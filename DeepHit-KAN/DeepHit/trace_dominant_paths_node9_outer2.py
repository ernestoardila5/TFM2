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
TARGET_CS_NODE = 9
EVAL_TIMES = [144, 288, 432]

BASE = Path("METABRIC/results_KAN1L_CTD")
FEATURE_FILE = Path("./sample data/METABRIC/cleaned_features_final.csv")
OUTDIR = Path("KAN1L_trace_dominant_paths_outer2_node9")


def load_normalizer(path):
    nf = np.load(path)
    return {
        "mode": str(nf["mode"]),
        "offset": nf["offset"],
        "scale": nf["scale"],
    }


def edge_activity(postacts, out_node):
    # postacts: [N, out_dim, in_dim]
    e = postacts[:, out_node, :]
    return torch.std(e, dim=0, unbiased=False), e


def print_edge_formula(title, layer, in_idx, out_idx, var_name,
                       offset=None, scale=None, raw_values=None):
    k = int(layer.k)
    mask = float(layer.mask[in_idx, out_idx].detach())
    sb = float(layer.scale_base[in_idx, out_idx].detach())
    ss = float(layer.scale_sp[in_idx, out_idx].detach())
    coef = layer.coef[in_idx, out_idx, :].detach().cpu().numpy()
    full_grid = layer.grid[in_idx, :].detach().cpu().numpy()
    core_grid = full_grid[k:-k] if k > 0 else full_grid.copy()

    print("\n" + "-" * 100)
    print(title)
    print("-" * 100)
    print(f"input index = {in_idx}")
    print(f"output node = {out_idx}")
    print(f"variable    = {var_name}")
    print(f"k           = {k}")
    print(f"mask        = {mask:.10f}")
    print(f"scale_base  = {sb:.10f}")
    print(f"scale_sp    = {ss:.10f}")
    print("base_fun    = SiLU")
    print("\nEXACT EDGE FORM")
    print(
        f"phi({var_name}) = {mask:.10f} * ["
        f"{sb:.10f}*SiLU({var_name}) + "
        f"{ss:.10f}*SUM_r c_r B_(r,{k})({var_name})]"
    )

    print("\nB-SPLINE COEFFICIENTS")
    for r, c in enumerate(coef):
        print(f"c[{r:02d}] = {c:+.10f}")

    print("\nCORE KNOTS (MODEL INPUT SCALE)")
    print(np.array2string(core_grid, precision=8, separator=", "))

    if offset is not None and scale is not None:
        raw_knots = core_grid * float(scale) + float(offset)
        print("\nCORE KNOTS (PRE-NORMALIZATION SCALE)")
        print(np.array2string(raw_knots, precision=8, separator=", "))
        print(
            f"\nNormalization: u = (x - ({float(offset):.10f})) "
            f"/ ({float(scale):.10f})"
        )

    if raw_values is not None:
        uniq = np.unique(raw_values)
        print(f"\nObserved unique raw values: {len(uniq)}")
        if len(uniq) <= 20:
            print(np.array2string(uniq, precision=8, separator=", "))
            print(
                "NOTE: this processed input is discrete/low-cardinality; "
                "a local derivative w.r.t. its continuous encoding should "
                "not be interpreted as a real infinitesimal clinical change."
            )
        else:
            print(
                f"min={np.nanmin(raw_values):.6f} | "
                f"max={np.nanmax(raw_values):.6f}"
            )

    return {
        "k": k,
        "mask": mask,
        "scale_base": sb,
        "scale_sp": ss,
        "coef": coef,
        "core_grid": core_grid,
    }


def main():
    OUTDIR.mkdir(exist_ok=True)

    df = pd.read_csv(FEATURE_FILE)
    feature_names = list(df.columns)
    raw_data = df.to_numpy(dtype=float)

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

    Xn = impt.apply_normalizer(raw_data, normalizer)
    Xdev = torch.tensor(Xn[dev_idx], dtype=torch.float32)

    # ========================================================
    # Forward to shared and CS layers
    # ========================================================
    with torch.no_grad():
        shared_out = model.shared_kan(Xdev)
        q = torch.cat([Xdev, shared_out], dim=1)

    shared = model.shared_kan
    cs = model.cs_kans[0]

    if len(shared.act_fun) != 1 or len(cs.act_fun) != 1:
        raise RuntimeError("This script is for the 1-layer OUTER-2 KANs.")

    shared_layer = shared.act_fun[0]
    cs_layer = cs.act_fun[0]

    with torch.no_grad():
        sh_xnum, _, sh_postacts, _ = shared_layer(Xdev)
        cs_xnum, _, cs_postacts, _ = cs_layer(q)
        c = cs(q)

    # ========================================================
    # Identify dominant direct and shared incoming edges to c9
    # ========================================================
    cs_activity, cs_edges = edge_activity(cs_postacts, TARGET_CS_NODE)

    n_original = Xdev.shape[1]
    n_shared = shared_out.shape[1]

    direct_activity = cs_activity[:n_original]
    shared_activity = cs_activity[n_original:]

    top_direct_idx = int(torch.argmax(direct_activity))
    top_shared_idx = int(torch.argmax(shared_activity))
    top_shared_q_idx = n_original + top_shared_idx

    direct_name = feature_names[top_direct_idx]
    shared_name = f"shared_output_{top_shared_idx}"

    print("\n" + "=" * 100)
    print("TRACE DOMINANT PATHS INTO CS NODE 9 — OUTER 2")
    print("=" * 100)
    print(f"Development patients: {len(Xdev)}")
    print(f"Top direct input -> c9: {direct_name} (index {top_direct_idx})")
    print(
        f"  activity SD = "
        f"{float(direct_activity[top_direct_idx]):.10f}"
    )
    print(
        f"Top shared input -> c9: {shared_name} "
        f"(q index {top_shared_q_idx})"
    )
    print(
        f"  activity SD = "
        f"{float(shared_activity[top_shared_idx]):.10f}"
    )

    # ========================================================
    # Exact formula: top direct feature -> c9
    # ========================================================
    direct_params = print_edge_formula(
        "PATH A — TOP DIRECT EDGE INTO c9",
        cs_layer,
        top_direct_idx,
        TARGET_CS_NODE,
        f"u_{direct_name}",
        offset=normalizer["offset"][top_direct_idx],
        scale=normalizer["scale"][top_direct_idx],
        raw_values=raw_data[dev_idx, top_direct_idx],
    )

    # ========================================================
    # Exact formula: shared_output_8 -> c9
    # ========================================================
    shared_to_c9_params = print_edge_formula(
        "PATH B, STEP 2 — TOP SHARED EDGE INTO c9",
        cs_layer,
        top_shared_q_idx,
        TARGET_CS_NODE,
        f"s_{top_shared_idx}",
    )

    # ========================================================
    # Open the dominant shared node itself
    # ========================================================
    sh_activity, sh_edges = edge_activity(sh_postacts, top_shared_idx)
    top_into_shared = torch.argsort(sh_activity, descending=True)

    print("\n" + "-" * 100)
    print(f"PATH B, STEP 1 — WHAT BUILDS shared_output_{top_shared_idx}?")
    print("-" * 100)
    print("TOP 15 ORIGINAL INPUTS INTO THIS SHARED NODE")

    rows = []
    for rank, idx_t in enumerate(top_into_shared[:15], start=1):
        idx = int(idx_t)
        rows.append({
            "rank": rank,
            "input_index": idx,
            "input_name": feature_names[idx],
            "activity_sd": float(sh_activity[idx]),
            "mean_abs_edge_output": float(
                torch.mean(torch.abs(sh_edges[:, idx]))
            ),
            "mask": float(shared_layer.mask[idx, top_shared_idx].detach()),
        })

    ranking_df = pd.DataFrame(rows)
    print(ranking_df.to_string(index=False))

    top_shared_source_idx = int(top_into_shared[0])
    top_shared_source_name = feature_names[top_shared_source_idx]

    # Exact node reconstruction for shared node
    sh_edge_sum = torch.sum(sh_edges, dim=1)

    sh_sub_scale = float(
        shared.subnode_scale[0][top_shared_idx].detach()
    )
    sh_sub_bias = float(
        shared.subnode_bias[0][top_shared_idx].detach()
    )
    sh_node_scale = float(
        shared.node_scale[0][top_shared_idx].detach()
    )
    sh_node_bias = float(
        shared.node_bias[0][top_shared_idx].detach()
    )

    sh_after_sub = (
        shared.subnode_scale[0][top_shared_idx].detach() * sh_edge_sum
        + shared.subnode_bias[0][top_shared_idx].detach()
    )
    sh_reconstructed = (
        shared.node_scale[0][top_shared_idx].detach() * sh_after_sub
        + shared.node_bias[0][top_shared_idx].detach()
    )
    sh_actual = shared_out[:, top_shared_idx]
    sh_err = torch.abs(sh_reconstructed - sh_actual)

    print("\nSHARED NODE AFFINE PARAMETERS")
    print(f"subnode_scale = {sh_sub_scale:.10f}")
    print(f"subnode_bias  = {sh_sub_bias:.10f}")
    print(f"node_scale    = {sh_node_scale:.10f}")
    print(f"node_bias     = {sh_node_bias:.10f}")

    print("\nSHARED NODE EXACT RECONSTRUCTION")
    print(f"max error  = {float(sh_err.max()):.12e}")
    print(f"mean error = {float(sh_err.mean()):.12e}")

    top_source_params = print_edge_formula(
        f"PATH B, STEP 1 — TOP EDGE INTO shared_output_{top_shared_idx}",
        shared_layer,
        top_shared_source_idx,
        top_shared_idx,
        f"u_{top_shared_source_name}",
        offset=normalizer["offset"][top_shared_source_idx],
        scale=normalizer["scale"][top_shared_source_idx],
        raw_values=raw_data[dev_idx, top_shared_source_idx],
    )

    # ========================================================
    # Local derivatives for the two explicit paths
    # ========================================================

    # A) direct feature -> c9
    q_req = q.detach().clone().requires_grad_(True)
    _, _, cs_post_req, _ = cs_layer(q_req)
    edge_direct = cs_post_req[:, TARGET_CS_NODE, top_direct_idx]
    dphi_direct_dq = torch.autograd.grad(
        edge_direct.sum(),
        q_req,
        retain_graph=False,
    )[0][:, top_direct_idx].detach()

    cs_affine_factor = (
        float(cs.subnode_scale[0][TARGET_CS_NODE].detach())
        * float(cs.node_scale[0][TARGET_CS_NODE].detach())
    )
    dc9_du_direct = cs_affine_factor * dphi_direct_dq

    # B) top source -> shared node -> c9
    X_req = Xdev.detach().clone().requires_grad_(True)
    _, _, sh_post_req, _ = shared_layer(X_req)
    edge_x_to_s = sh_post_req[
        :, top_shared_idx, top_shared_source_idx
    ]
    dphi_x_to_s_du = torch.autograd.grad(
        edge_x_to_s.sum(),
        X_req,
        retain_graph=False,
    )[0][:, top_shared_source_idx].detach()

    sh_affine_factor = (
        float(shared.subnode_scale[0][top_shared_idx].detach())
        * float(shared.node_scale[0][top_shared_idx].detach())
    )
    ds_du_path = sh_affine_factor * dphi_x_to_s_du

    q2_req = q.detach().clone().requires_grad_(True)
    _, _, cs_post_req2, _ = cs_layer(q2_req)
    edge_s_to_c = cs_post_req2[
        :, TARGET_CS_NODE, top_shared_q_idx
    ]
    dphi_s_to_c_ds = torch.autograd.grad(
        edge_s_to_c.sum(),
        q2_req,
        retain_graph=False,
    )[0][:, top_shared_q_idx].detach()

    dc9_ds_path = cs_affine_factor * dphi_s_to_c_ds
    dc9_du_shared_path = dc9_ds_path * ds_du_path

    # Downstream dF/dc9
    W = model.output_layer.weight.detach()
    b = model.output_layer.bias.detach()

    with torch.no_grad():
        logits = c @ W.T + b[None, :]
        p = torch.softmax(logits, dim=1)

    path_rows = []

    for h in EVAL_TIMES:
        with torch.no_grad():
            Fh = p[:, :h+1].sum(dim=1)
            dF_dc9 = (
                p[:, :h+1] @ W[:h+1, TARGET_CS_NODE]
                - Fh * (p @ W[:, TARGET_CS_NODE])
            )

            # Sensitivity per normalized feature unit
            dF_du_direct_path = dF_dc9 * dc9_du_direct
            dF_du_shared_path = dF_dc9 * dc9_du_shared_path

            # Convert to pre-normalization feature scale.
            direct_scale = float(normalizer["scale"][top_direct_idx])
            source_scale = float(normalizer["scale"][top_shared_source_idx])

            dF_dx_direct_path = dF_du_direct_path / direct_scale
            dF_dx_shared_path = dF_du_shared_path / source_scale

        for path_name, arr, feature_name in [
            (
                "direct_to_c9",
                dF_dx_direct_path,
                direct_name,
            ),
            (
                f"via_shared_{top_shared_idx}_to_c9",
                dF_dx_shared_path,
                top_shared_source_name,
            ),
        ]:
            path_rows.append({
                "time_index": h,
                "path": path_name,
                "source_feature": feature_name,
                "RMS_dF_dx_path": float(
                    torch.sqrt(torch.mean(arr ** 2))
                ),
                "mean_abs_dF_dx_path": float(
                    torch.mean(torch.abs(arr))
                ),
                "mean_dF_dx_path": float(torch.mean(arr)),
                "positive_fraction": float(
                    torch.mean((arr > 0).float())
                ),
            })

    path_df = pd.DataFrame(path_rows)

    print("\n" + "-" * 100)
    print("LOCAL CIF SENSITIVITY ALONG THE TWO EXPLICIT PATHS")
    print("-" * 100)
    print(
        "These are PATH-SPECIFIC derivatives, not the total effect "
        "of the source feature through the whole network."
    )
    print(path_df.to_string(index=False))

    # ========================================================
    # Save outputs
    # ========================================================
    ranking_df.to_csv(
        OUTDIR / f"shared_output_{top_shared_idx}_incoming_ranked.csv",
        index=False,
    )
    path_df.to_csv(
        OUTDIR / "dominant_path_CIF_sensitivities.csv",
        index=False,
    )

    summary = f"""OUTER 2 — DOMINANT PATH TRACE INTO CS NODE {TARGET_CS_NODE}

Top direct path:
    {direct_name} -> c{TARGET_CS_NODE} -> CIF

Top shared-input path into c{TARGET_CS_NODE}:
    {top_shared_source_name}
      -> shared_output_{top_shared_idx}
      -> c{TARGET_CS_NODE}
      -> CIF

Important:
    The path-specific derivatives reported here are not the total
    derivative of the source feature with respect to CIF, because a
    source feature can reach the output through many other shared and
    CS nodes simultaneously.
"""
    (OUTDIR / "path_summary.txt").write_text(summary, encoding="utf-8")

    print("\nFILES")
    print(OUTDIR / f"shared_output_{top_shared_idx}_incoming_ranked.csv")
    print(OUTDIR / "dominant_path_CIF_sensitivities.csv")
    print(OUTDIR / "path_summary.txt")
    print("=" * 100)


if __name__ == "__main__":
    main()
