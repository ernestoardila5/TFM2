from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

import import_data as impt
from class_DeepHit import Model_DeepHit
from get_main_CTD_1L import get_network_settings


OUTER = 2
TARGET_NODE = 9
EVAL_TIMES = [144, 288, 432]

BASE = Path("METABRIC/results_KAN1L_CTD")
FEATURE_FILE = Path("./sample data/METABRIC/cleaned_features_final.csv")
OUTDIR = Path("KAN1L_exact_node_outer2_node9")


def load_normalizer(path):
    nf = np.load(path)
    return {
        "mode": str(nf["mode"]),
        "offset": nf["offset"],
        "scale": nf["scale"],
    }


def main():
    OUTDIR.mkdir(exist_ok=True)

    # --------------------------------------------------------
    # Data and exact OUTER-2 development split
    # --------------------------------------------------------
    df = pd.read_csv(FEATURE_FILE)
    feature_names = list(df.columns)
    raw_data = df.to_numpy(dtype=float)

    x_dim, DATA, MASK = impt.import_dataset_METABRIC()
    mask1, _ = MASK

    if len(feature_names) != x_dim:
        raise RuntimeError(
            f"Feature-name count ({len(feature_names)}) != x_dim ({x_dim})"
        )

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

    if input_dims["num_Event"] != 1:
        raise RuntimeError("This script assumes single-event METABRIC.")

    model = Model_DeepHit(
        input_dims,
        get_network_settings(hp, seed=1234 + OUTER),
    )
    model.load_state_dict(
        torch.load(model_path, map_location="cpu")
    )
    model.eval()

    Xn = impt.apply_normalizer(raw_data, normalizer)
    Xdev = torch.tensor(Xn[dev_idx], dtype=torch.float32)

    # --------------------------------------------------------
    # Shared representation and CS input q=[x, shared(x)]
    # --------------------------------------------------------
    with torch.no_grad():
        shared = model.shared_kan(Xdev)
        q = torch.cat([Xdev, shared], dim=1)

    n_original = Xdev.shape[1]
    n_shared = shared.shape[1]
    n_cs_inputs = q.shape[1]

    input_names = (
        feature_names
        + [f"shared_output_{j}" for j in range(n_shared)]
    )
    input_types = (
        ["original_feature"] * n_original
        + ["shared_output"] * n_shared
    )

    if len(input_names) != n_cs_inputs:
        raise RuntimeError("CS input-name dimension mismatch.")

    cs = model.cs_kans[0]

    if len(cs.act_fun) != 1:
        raise RuntimeError(
            f"Expected 1-layer CS-KAN, got {len(cs.act_fun)} KAN layers."
        )

    if getattr(cs, "symbolic_enabled", False):
        raise RuntimeError(
            "This exact numerical reconstruction assumes symbolic_enabled=False."
        )

    layer = cs.act_fun[0]

    # --------------------------------------------------------
    # Exact numerical edge outputs
    #
    # KANLayer.forward returns:
    #   x_numerical: [N, out_dim] = sum of incoming edge outputs
    #   postacts:    [N, out_dim, in_dim] = each phi_ij(q_i)
    # --------------------------------------------------------
    with torch.no_grad():
        x_num, preacts, postacts, postspline = layer(q)
        c_full = cs(q)

    if TARGET_NODE >= x_num.shape[1]:
        raise RuntimeError(
            f"Target node {TARGET_NODE} outside CS output dimension "
            f"{x_num.shape[1]}"
        )

    edge_out = postacts[:, TARGET_NODE, :]  # [N, 143]
    summed_from_edges = torch.sum(edge_out, dim=1)
    summed_layer = x_num[:, TARGET_NODE]

    # --------------------------------------------------------
    # Reconstruct the affine stages used by MultKAN.
    # OUTER 2 CS-KAN has additive nodes only, so no
    # multiplicative combination is expected here.
    # --------------------------------------------------------
    sub_scale = float(cs.subnode_scale[0][TARGET_NODE].detach())
    sub_bias = float(cs.subnode_bias[0][TARGET_NODE].detach())
    node_scale = float(cs.node_scale[0][TARGET_NODE].detach())
    node_bias = float(cs.node_bias[0][TARGET_NODE].detach())

    after_sub = (
        cs.subnode_scale[0][TARGET_NODE].detach()
        * summed_from_edges
        + cs.subnode_bias[0][TARGET_NODE].detach()
    )

    reconstructed_final = (
        cs.node_scale[0][TARGET_NODE].detach()
        * after_sub
        + cs.node_bias[0][TARGET_NODE].detach()
    )

    actual_final = c_full[:, TARGET_NODE]

    edge_sum_err = torch.abs(summed_from_edges - summed_layer)
    final_err = torch.abs(reconstructed_final - actual_final)

    # --------------------------------------------------------
    # Incoming-edge activity on real development patients
    # --------------------------------------------------------
    activity_sd = torch.std(
        edge_out,
        dim=0,
        unbiased=False,
    )
    mean_abs = torch.mean(
        torch.abs(edge_out),
        dim=0,
    )
    mean_edge = torch.mean(
        edge_out,
        dim=0,
    )

    edge_table = pd.DataFrame({
        "input_index": np.arange(n_cs_inputs),
        "input_name": input_names,
        "input_type": input_types,
        "activity_sd": activity_sd.cpu().numpy(),
        "mean_abs_edge_output": mean_abs.cpu().numpy(),
        "mean_edge_output": mean_edge.cpu().numpy(),
        "mask": layer.mask[:, TARGET_NODE].detach().cpu().numpy(),
    })

    edge_table = edge_table.sort_values(
        "activity_sd",
        ascending=False,
    ).reset_index(drop=True)
    edge_table.insert(
        0,
        "activity_rank",
        np.arange(1, len(edge_table) + 1),
    )

    source_summary = (
        edge_table
        .groupby("input_type", as_index=False)
        .agg(
            n_edges=("input_index", "count"),
            sum_activity_sd=("activity_sd", "sum"),
            mean_activity_sd=("activity_sd", "mean"),
            max_activity_sd=("activity_sd", "max"),
            sum_mean_abs=("mean_abs_edge_output", "sum"),
        )
    )

    # Exact additive aggregates for this node
    direct_sum = torch.sum(edge_out[:, :n_original], dim=1)
    shared_sum = torch.sum(edge_out[:, n_original:], dim=1)

    aggregate_summary = pd.DataFrame([
        {
            "component": "direct_original_features",
            "mean": float(direct_sum.mean()),
            "sd": float(torch.std(direct_sum, unbiased=False)),
            "mean_abs": float(torch.mean(torch.abs(direct_sum))),
        },
        {
            "component": "shared_outputs",
            "mean": float(shared_sum.mean()),
            "sd": float(torch.std(shared_sum, unbiased=False)),
            "mean_abs": float(torch.mean(torch.abs(shared_sum))),
        },
        {
            "component": "pre_affine_total",
            "mean": float(summed_from_edges.mean()),
            "sd": float(torch.std(summed_from_edges, unbiased=False)),
            "mean_abs": float(torch.mean(torch.abs(summed_from_edges))),
        },
        {
            "component": "final_node",
            "mean": float(actual_final.mean()),
            "sd": float(torch.std(actual_final, unbiased=False)),
            "mean_abs": float(torch.mean(torch.abs(actual_final))),
        },
    ])

    # --------------------------------------------------------
    # Downstream CIF sensitivity of node 9
    # --------------------------------------------------------
    W = model.output_layer.weight.detach()
    b = model.output_layer.bias.detach()

    with torch.no_grad():
        logits = c_full @ W.T + b[None, :]
        p = torch.softmax(logits, dim=1)

    node_sd = float(
        torch.std(actual_final, unbiased=False)
    )

    sensitivity_rows = []

    for h in EVAL_TIMES:
        if h >= W.shape[0]:
            continue

        with torch.no_grad():
            Fh = torch.sum(p[:, :h + 1], dim=1)

            first_term_j = (
                p[:, :h + 1]
                @ W[:h + 1, TARGET_NODE]
            )

            expected_W_j = (
                p @ W[:, TARGET_NODE]
            )

            grad = (
                first_term_j
                - Fh * expected_W_j
            )

            rms = torch.sqrt(torch.mean(grad ** 2))
            mean_abs_grad = torch.mean(torch.abs(grad))
            mean_grad = torch.mean(grad)
            pos_frac = torch.mean((grad > 0).float())

        sensitivity_rows.append({
            "time_index": h,
            "node": TARGET_NODE,
            "node_sd": node_sd,
            "dCIF_dc_rms": float(rms),
            "dCIF_dc_mean_abs": float(mean_abs_grad),
            "dCIF_dc_mean": float(mean_grad),
            "positive_gradient_fraction": float(pos_frac),
            "standardized_sensitivity_rms": node_sd * float(rms),
        })

    sensitivity_df = pd.DataFrame(sensitivity_rows)

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------
    print("\n" + "=" * 100)
    print("EXACT CS-KAN NODE 9 ANALYSIS — OUTER 2")
    print("=" * 100)
    print(f"Development patients: {len(Xdev)}")
    print(f"Original features: {n_original}")
    print(f"Shared outputs: {n_shared}")
    print(f"Total CS inputs: {n_cs_inputs}")
    print(f"CS outputs: {c_full.shape[1]}")
    print(f"Target node: {TARGET_NODE}")
    print(f"k = {layer.k}")

    print("\nTOP 20 INCOMING EDGES BY ACTIVITY SD")
    cols = [
        "activity_rank",
        "input_index",
        "input_name",
        "input_type",
        "activity_sd",
        "mean_abs_edge_output",
        "mask",
    ]
    print(
        edge_table.loc[:, cols]
        .head(20)
        .to_string(index=False)
    )

    print("\nTOP 10 DIRECT ORIGINAL FEATURES")
    print(
        edge_table[
            edge_table["input_type"] == "original_feature"
        ].loc[:, cols]
        .head(10)
        .to_string(index=False)
    )

    print("\nTOP 10 SHARED OUTPUTS")
    print(
        edge_table[
            edge_table["input_type"] == "shared_output"
        ].loc[:, cols]
        .head(10)
        .to_string(index=False)
    )

    print("\nSOURCE-TYPE SUMMARY")
    print(source_summary.to_string(index=False))

    print("\nEXACT ADDITIVE AGGREGATES FOR NODE 9")
    print(aggregate_summary.to_string(index=False))

    print("\nNODE AFFINE PARAMETERS")
    print(f"subnode_scale = {sub_scale:.10f}")
    print(f"subnode_bias  = {sub_bias:.10f}")
    print(f"node_scale    = {node_scale:.10f}")
    print(f"node_bias     = {node_bias:.10f}")

    print("\nEXACT RECONSTRUCTION CHECK")
    print(
        "max incoming-edge sum error = "
        f"{float(edge_sum_err.max()):.12e}"
    )
    print(
        "mean incoming-edge sum error = "
        f"{float(edge_sum_err.mean()):.12e}"
    )
    print(
        "max FINAL node error = "
        f"{float(final_err.max()):.12e}"
    )
    print(
        "mean FINAL node error = "
        f"{float(final_err.mean()):.12e}"
    )

    print("\nDOWNSTREAM CIF SENSITIVITY OF NODE 9")
    print(sensitivity_df.to_string(index=False))

    # --------------------------------------------------------
    # Save files
    # --------------------------------------------------------
    edge_table.to_csv(
        OUTDIR / "node9_incoming_edges_ranked.csv",
        index=False,
    )
    source_summary.to_csv(
        OUTDIR / "node9_source_type_summary.csv",
        index=False,
    )
    aggregate_summary.to_csv(
        OUTDIR / "node9_direct_vs_shared_aggregate.csv",
        index=False,
    )
    sensitivity_df.to_csv(
        OUTDIR / "node9_CIF_sensitivity.csv",
        index=False,
    )

    eq_text = f"""EXACT OUTER-2 CS NODE {TARGET_NODE}

Inputs:
    q = [x_normalized, shared(x)]
    dim(q) = {n_cs_inputs}
    original inputs = {n_original}
    shared outputs = {n_shared}

Each numerical KAN edge is:

    phi_i,{TARGET_NODE}(q_i)
      = mask_i,{TARGET_NODE} * [
          scale_base_i,{TARGET_NODE} * SiLU(q_i)
          + scale_sp_i,{TARGET_NODE}
            * SUM_r c_(i,{TARGET_NODE},r) B_(r,k)(q_i)
        ]

The pre-affine additive node is:

    r_{TARGET_NODE}(q)
      = SUM_i phi_i,{TARGET_NODE}(q_i)

For this trained checkpoint:

    subnode transform:
      r_tilde = ({sub_scale}) * r_{TARGET_NODE} + ({sub_bias})

    final node transform:
      c_{TARGET_NODE} = ({node_scale}) * r_tilde + ({node_bias})

The CSV file node9_incoming_edges_ranked.csv gives the 143
incoming edge functions ranked by their observed activity SD
on the OUTER-2 development patients.

The file node9_CIF_sensitivity.csv gives the downstream local
sensitivity dF_h/dc_{TARGET_NODE} at h = 144, 288, 432.
"""
    (OUTDIR / "exact_node9_equation.txt").write_text(
        eq_text,
        encoding="utf-8",
    )

    print("\nFILES")
    for name in [
        "node9_incoming_edges_ranked.csv",
        "node9_source_type_summary.csv",
        "node9_direct_vs_shared_aggregate.csv",
        "node9_CIF_sensitivity.csv",
        "exact_node9_equation.txt",
    ]:
        print(OUTDIR / name)

    print("=" * 100)


if __name__ == "__main__":
    main()
