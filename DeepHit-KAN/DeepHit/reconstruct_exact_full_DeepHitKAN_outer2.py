from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

import import_data as impt
from class_DeepHit import Model_DeepHit
from get_main_CTD_1L import get_network_settings
from spline import B_batch


OUTER = 2
TARGET_NODE = 17
EVAL_TIMES = [144, 288, 432]

BASE = Path("METABRIC/results_KAN1L_CTD")
FEATURE_FILE = Path("./sample data/METABRIC/cleaned_features_final.csv")
OUTDIR = Path("KAN1L_exact_full_model_outer2")


def load_normalizer(path):
    nf = np.load(path)
    return {
        "mode": str(nf["mode"]),
        "offset": nf["offset"],
        "scale": nf["scale"],
    }


def reconstruct_one_layer_additive(multkan, x):
    """
    Exact numerical reconstruction of a 1-layer, additive-only MultKAN.
    Uses the stored grid, spline coefficients, masks and affine transforms.
    """
    if int(multkan.depth) != 1:
        raise RuntimeError(f"Expected depth=1, got {multkan.depth}")
    if bool(multkan.symbolic_enabled):
        raise RuntimeError("Expected symbolic_enabled=False")

    dim_sum = int(multkan.width[1][0])
    dim_mult = int(multkan.width[1][1])
    if dim_mult != 0:
        raise RuntimeError(
            f"This exact helper is additive-only, got "
            f"{dim_sum} additive + {dim_mult} multiplicative nodes"
        )

    # MultKAN.forward first selects input_id.
    x_sel = x[:, multkan.input_id.long()]
    layer = multkan.act_fun[0]

    with torch.no_grad():
        basis = B_batch(x_sel, layer.grid, k=int(layer.k))
        spline_values = torch.einsum(
            "nir,ior->nio",
            basis,
            layer.coef,
        )
        base_values = layer.base_fun(x_sel)

        # [N, in_dim, out_dim]
        edge_values = (
            layer.scale_base[None, :, :] * base_values[:, :, None]
            + layer.scale_sp[None, :, :] * spline_values
        )
        edge_values = layer.mask[None, :, :] * edge_values

        # Sum incoming edge functions -> subnodes
        summed = torch.sum(edge_values, dim=1)

        # Subnode affine transform
        subnodes = (
            multkan.subnode_scale[0][None, :] * summed
            + multkan.subnode_bias[0][None, :]
        )

        # No multiplicative nodes in Outer 2; therefore subnodes are nodes.
        nodes = (
            multkan.node_scale[0][None, :] * subnodes
            + multkan.node_bias[0][None, :]
        )

    return nodes, edge_values, summed, subnodes


def err_stats(a, b):
    e = torch.abs(a - b)
    return float(e.max()), float(e.mean())


def main():
    OUTDIR.mkdir(exist_ok=True)

    df = pd.read_csv(FEATURE_FILE)
    raw_data = df.to_numpy(dtype=float)
    feature_names = list(df.columns)

    x_dim, DATA, MASK = impt.import_dataset_METABRIC()
    mask1, _ = MASK

    all_idx = np.arange(len(raw_data))
    dev_idx, _ = train_test_split(
        all_idx,
        test_size=0.20,
        random_state=1234 + OUTER,
    )

    itr_dir = BASE / f"itr_{OUTER}"
    hp_path = itr_dir / "hyperparameters_log.json"
    model_path = itr_dir / "models" / f"model_itr_{OUTER}.pt"
    norm_path = itr_dir / "models" / f"normalizer_itr_{OUTER}.npz"

    for p in [hp_path, model_path, norm_path]:
        if not p.exists():
            raise FileNotFoundError(p)

    with hp_path.open("r", encoding="utf-8") as f:
        hp = json.load(f)

    normalizer = load_normalizer(norm_path)

    input_dims = {
        "x_dim": x_dim,
        "num_Event": mask1.shape[1],
        "num_Category": mask1.shape[2],
    }

    model = Model_DeepHit(
        input_dims,
        get_network_settings(hp, seed=1234 + OUTER),
    )
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    if input_dims["num_Event"] != 1:
        raise RuntimeError(
            "This script is written for the single-event METABRIC case."
        )

    Xn = impt.apply_normalizer(raw_data, normalizer)
    Xdev = torch.tensor(Xn[dev_idx], dtype=torch.float32)

    # ========================================================
    # A) SHARED KAN — exact manual reconstruction
    # ========================================================
    with torch.no_grad():
        shared_actual = model.shared_kan(Xdev)

    (
        shared_manual,
        shared_edges,
        shared_summed,
        shared_subnodes,
    ) = reconstruct_one_layer_additive(model.shared_kan, Xdev)

    shared_max, shared_mean = err_stats(shared_manual, shared_actual)

    # ========================================================
    # B) CS KAN — exact manual reconstruction, using MANUAL shared
    # ========================================================
    h_cs_manual = torch.cat([Xdev, shared_manual], dim=1)

    with torch.no_grad():
        # Reference uses the actual model path.
        h_cs_actual = torch.cat([Xdev, shared_actual], dim=1)
        cs_actual = model.cs_kans[0](h_cs_actual)

    (
        cs_manual,
        cs_edges,
        cs_summed,
        cs_subnodes,
    ) = reconstruct_one_layer_additive(model.cs_kans[0], h_cs_manual)

    cs_max, cs_mean = err_stats(cs_manual, cs_actual)

    # ========================================================
    # C) FINAL LINEAR LAYER — exact logits
    # ========================================================
    W = model.output_layer.weight.detach().cpu()
    b = model.output_layer.bias.detach().cpu()

    with torch.no_grad():
        logits_actual = model.output_layer(cs_actual)
        logits_manual = cs_manual @ W.T + b[None, :]

    logits_max, logits_mean = err_stats(logits_manual, logits_actual)

    # ========================================================
    # D) SOFTMAX — exact event-time PMF
    # ========================================================
    with torch.no_grad():
        probs_manual_flat = torch.softmax(logits_manual, dim=1)
        probs_manual = probs_manual_flat.view(
            -1,
            input_dims["num_Event"],
            input_dims["num_Category"],
        )
        probs_actual = model(Xdev)

    probs_max, probs_mean = err_stats(probs_manual, probs_actual)

    # ========================================================
    # E) CIF — cumulative probability
    # ========================================================
    cif_manual = torch.cumsum(probs_manual[:, 0, :], dim=1)
    cif_actual = torch.cumsum(probs_actual[:, 0, :], dim=1)
    cif_max, cif_mean = err_stats(cif_manual, cif_actual)

    # ========================================================
    # F) Analyze final Linear contributions of all 32 CS nodes
    # ========================================================
    # contribution[n,t,j] = c_j(x_n) * W[t,j]
    contributions = (
        cs_actual[:, None, :] * W[None, :, :]
    )

    node_rms = torch.sqrt(
        torch.mean(contributions ** 2, dim=(0, 1))
    ).cpu().numpy()

    node_mean_abs = torch.mean(
        torch.abs(contributions),
        dim=(0, 1),
    ).cpu().numpy()

    node_sd = torch.std(
        contributions.reshape(-1, contributions.shape[2]),
        dim=0,
        unbiased=False,
    ).cpu().numpy()

    node_rank = pd.DataFrame({
        "node": np.arange(W.shape[1]),
        "global_logit_contribution_rms": node_rms,
        "global_logit_contribution_mean_abs": node_mean_abs,
        "global_logit_contribution_sd": node_sd,
    }).sort_values(
        "global_logit_contribution_rms",
        ascending=False,
    ).reset_index(drop=True)

    node_rank.insert(
        0,
        "global_rank",
        np.arange(1, len(node_rank) + 1),
    )
    node_rank.to_csv(
        OUTDIR / "cs_nodes_ranked_by_final_logit_contribution.csv",
        index=False,
    )

    # Long-form final output weights
    weight_rows = []
    for t in range(W.shape[0]):
        for j in range(W.shape[1]):
            weight_rows.append({
                "time_index": t,
                "cs_node": j,
                "weight_W_tj": float(W[t, j]),
                "bias_b_t": float(b[t]),
            })

    pd.DataFrame(weight_rows).to_csv(
        OUTDIR / "final_linear_weights_long.csv",
        index=False,
    )

    # Horizon-specific ranking and node 17
    horizon_lines = []
    horizon_tables = []

    for t in EVAL_TIMES:
        if t >= W.shape[0]:
            horizon_lines.append(
                f"time {t}: outside output dimension {W.shape[0]}"
            )
            continue

        contrib_t = cs_actual * W[t, :][None, :]
        mean_abs_t = torch.mean(
            torch.abs(contrib_t), dim=0
        ).cpu().numpy()
        sd_t = torch.std(
            contrib_t, dim=0, unbiased=False
        ).cpu().numpy()

        tab = pd.DataFrame({
            "time_index": t,
            "node": np.arange(W.shape[1]),
            "W_tj": W[t, :].cpu().numpy(),
            "mean_abs_contribution": mean_abs_t,
            "sd_contribution": sd_t,
        }).sort_values(
            "mean_abs_contribution",
            ascending=False,
        ).reset_index(drop=True)

        tab.insert(
            1,
            "rank_at_time",
            np.arange(1, len(tab) + 1),
        )
        horizon_tables.append(tab)

        row17 = tab[tab["node"] == TARGET_NODE].iloc[0]
        horizon_lines.append(
            f"time {t}: node {TARGET_NODE} | "
            f"W={row17['W_tj']:+.10f} | "
            f"rank={int(row17['rank_at_time'])}/{W.shape[1]} | "
            f"mean_abs_contribution="
            f"{row17['mean_abs_contribution']:.10f} | "
            f"sd={row17['sd_contribution']:.10f}"
        )

    if horizon_tables:
        pd.concat(horizon_tables, ignore_index=True).to_csv(
            OUTDIR / "horizon_node_contributions.csv",
            index=False,
        )

    # ========================================================
    # G) Save exact hierarchical equation
    # ========================================================
    s_dim = shared_manual.shape[1]
    c_dim = cs_manual.shape[1]
    T = W.shape[0]

    equation = f"""EXACT HIERARCHICAL DEEPHIT-KAN EQUATION — OUTER {OUTER}
================================================================================

INPUT NORMALIZATION
For each original feature i:
    u_i = (x_i - offset_i) / scale_i

SHARED KAN
There are {s_dim} additive Shared-KAN outputs and no multiplication nodes.

For j = 0,...,{s_dim-1}:
    r^S_j(u) = SUM_i phi^S_(i,j)(u_i)

    phi^S_(i,j)(v)
      = mask^S_(i,j) * [
            scale_base^S_(i,j) * SiLU(v)
          + scale_sp^S_(i,j) * SUM_r c^S_(i,j,r) B^S_(i,r,k)(v)
        ]

    subnode^S_j = subnode_scale^S_j * r^S_j + subnode_bias^S_j
    s_j          = node_scale^S_j * subnode^S_j + node_bias^S_j

CAUSE-SPECIFIC INPUT
    q = [u_0,...,u_{x_dim-1}, s_0,...,s_{s_dim-1}]

CAUSE-SPECIFIC KAN
There are {c_dim} additive CS-KAN outputs and no multiplication nodes.

For m = 0,...,{c_dim-1}:
    r^C_m(q) = SUM_i phi^C_(i,m)(q_i)

    phi^C_(i,m)(v)
      = mask^C_(i,m) * [
            scale_base^C_(i,m) * SiLU(v)
          + scale_sp^C_(i,m) * SUM_r c^C_(i,m,r) B^C_(i,r,k)(v)
        ]

    subnode^C_m = subnode_scale^C_m * r^C_m + subnode_bias^C_m
    c_m          = node_scale^C_m * subnode^C_m + node_bias^C_m

FINAL LINEAR LAYER
For each discrete time t = 0,...,{T-1}:
    z_t(x) = b_t + SUM_(m=0)^{c_dim-1} W_(t,m) c_m(x)

SOFTMAX — SINGLE EVENT METABRIC
    p_t(x) = exp(z_t(x)) / SUM_(s=0)^{T-1} exp(z_s(x))

CIF
    F(t|x) = SUM_(tau=0)^t p_tau(x)

NUMERICAL RECONSTRUCTION ERRORS
    Shared KAN max error = {shared_max:.12e}
    Shared KAN mean error = {shared_mean:.12e}

    CS KAN max error = {cs_max:.12e}
    CS KAN mean error = {cs_mean:.12e}

    Logits max error = {logits_max:.12e}
    Logits mean error = {logits_mean:.12e}

    Softmax probabilities max error = {probs_max:.12e}
    Softmax probabilities mean error = {probs_mean:.12e}

    CIF max error = {cif_max:.12e}
    CIF mean error = {cif_mean:.12e}

NODE {TARGET_NODE} IN FINAL LINEAR LAYER
{chr(10).join(horizon_lines)}
"""

    (OUTDIR / "exact_full_model_equation.txt").write_text(
        equation,
        encoding="utf-8",
    )

    # ========================================================
    # H) Console
    # ========================================================
    print("\n" + "=" * 100)
    print("FULL EXACT DEEPHIT-KAN RECONSTRUCTION — OUTER 2")
    print("=" * 100)
    print(f"Shared outputs: {s_dim}")
    print(f"CS outputs: {c_dim}")
    print(f"Discrete time logits: {T}")

    print("\nEXACT RECONSTRUCTION ERRORS")
    print(f"Shared KAN max   = {shared_max:.12e}")
    print(f"Shared KAN mean  = {shared_mean:.12e}")
    print(f"CS KAN max       = {cs_max:.12e}")
    print(f"CS KAN mean      = {cs_mean:.12e}")
    print(f"Logits max       = {logits_max:.12e}")
    print(f"Logits mean      = {logits_mean:.12e}")
    print(f"Probabilities max= {probs_max:.12e}")
    print(f"Probabilities mean={probs_mean:.12e}")
    print(f"CIF max          = {cif_max:.12e}")
    print(f"CIF mean         = {cif_mean:.12e}")

    print("\nTOP 10 CS NODES BY GLOBAL DIRECT CONTRIBUTION TO LOGITS")
    print(
        node_rank.head(10).to_string(index=False)
    )

    row17_global = node_rank[
        node_rank["node"] == TARGET_NODE
    ].iloc[0]
    print(f"\nNODE {TARGET_NODE} GLOBAL FINAL-LAYER RELEVANCE")
    print(
        f"rank={int(row17_global['global_rank'])}/{W.shape[1]} | "
        f"RMS={row17_global['global_logit_contribution_rms']:.10f} | "
        f"mean_abs={row17_global['global_logit_contribution_mean_abs']:.10f}"
    )

    print("\nNODE 17 AT SELECTED TIME INDICES")
    for line in horizon_lines:
        print(line)

    print("\nFILES")
    print(OUTDIR / "exact_full_model_equation.txt")
    print(OUTDIR / "cs_nodes_ranked_by_final_logit_contribution.csv")
    print(OUTDIR / "final_linear_weights_long.csv")
    print(OUTDIR / "horizon_node_contributions.csv")
    print("=" * 100)


if __name__ == "__main__":
    main()
