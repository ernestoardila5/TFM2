import json
from pathlib import Path

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
BASE = Path("METABRIC/results_KAN1L_CTD")
FEATURE_FILE = Path("./sample data/METABRIC/cleaned_features_final.csv")
OUTDIR = Path("KAN1L_exact_node_outer2_node17")


def load_normalizer(path):
    nf = np.load(path)
    return {
        "mode": str(nf["mode"]),
        "offset": nf["offset"],
        "scale": nf["scale"],
    }


def arr_json(a):
    return json.dumps([float(v) for v in np.asarray(a).ravel()])


def fmt(x):
    return f"{float(x):+.10g}"


def main():
    OUTDIR.mkdir(exist_ok=True)

    df = pd.read_csv(FEATURE_FILE)
    feature_names = list(df.columns)
    raw_data = df.to_numpy(dtype=float)

    x_dim, DATA, MASK = impt.import_dataset_METABRIC()
    mask1, _ = MASK

    all_idx = np.arange(len(raw_data))
    dev_idx, _ = train_test_split(
        all_idx, test_size=0.20, random_state=1234 + OUTER
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

    with torch.no_grad():
        shared_dev = model.shared_kan(Xdev)
        h_cs_dev = torch.cat([Xdev, shared_dev], dim=1)

    cs_kan = model.cs_kans[0]

    if int(cs_kan.depth) != 1:
        raise RuntimeError(f"Expected 1-layer CS-KAN, got depth={cs_kan.depth}")

    if bool(cs_kan.symbolic_enabled):
        raise RuntimeError("symbolic_enabled=True; this script expects numerical KAN only")

    layer = cs_kan.act_fun[0]

    num_original = x_dim
    num_shared = int(shared_dev.shape[1])
    num_cs_inputs = int(h_cs_dev.shape[1])

    input_names = feature_names + [f"shared_{j:02d}" for j in range(num_shared)]
    input_types = ["original_feature"] * num_original + ["shared_output"] * num_shared

    dim_sum = int(cs_kan.width[1][0])
    dim_mult = int(cs_kan.width[1][1])

    if TARGET_NODE >= dim_sum:
        raise RuntimeError(
            f"TARGET_NODE={TARGET_NODE} is not additive: "
            f"dim_sum={dim_sum}, dim_mult={dim_mult}"
        )

    with torch.no_grad():
        layer_y, preacts, postacts, postspline = layer(h_cs_dev)
        cs_out = cs_kan(h_cs_dev)

        actual_edges = postacts[:, TARGET_NODE, :]
        actual_layer_node = layer_y[:, TARGET_NODE]

        basis = B_batch(h_cs_dev, layer.grid, k=int(layer.k))
        spline_values = torch.einsum("nir,ior->nio", basis, layer.coef)
        base_values = layer.base_fun(h_cs_dev)

        edge_manual = (
            layer.scale_base[None, :, :] * base_values[:, :, None]
            + layer.scale_sp[None, :, :] * spline_values
        )
        edge_manual = layer.mask[None, :, :] * edge_manual

        manual_edges_target = edge_manual[:, :, TARGET_NODE]
        manual_layer_node = torch.sum(manual_edges_target, dim=1)

        subnode_value = (
            cs_kan.subnode_scale[0][TARGET_NODE] * manual_layer_node
            + cs_kan.subnode_bias[0][TARGET_NODE]
        )
        manual_final_node = (
            cs_kan.node_scale[0][TARGET_NODE] * subnode_value
            + cs_kan.node_bias[0][TARGET_NODE]
        )
        actual_final_node = cs_out[:, TARGET_NODE]

    edge_err = torch.abs(manual_edges_target - actual_edges)
    layer_node_err = torch.abs(manual_layer_node - actual_layer_node)
    final_node_err = torch.abs(manual_final_node - actual_final_node)

    activity_sd = torch.std(actual_edges, dim=0, unbiased=False).cpu().numpy()
    mean_abs = torch.mean(torch.abs(actual_edges), dim=0).cpu().numpy()
    mean_signed = torch.mean(actual_edges, dim=0).cpu().numpy()

    rows = []
    k = int(layer.k)

    for i in range(num_cs_inputs):
        coef = layer.coef[i, TARGET_NODE, :].detach().cpu().numpy().astype(float)
        grid = layer.grid[i].detach().cpu().numpy().astype(float)

        if i < num_original:
            off = float(normalizer["offset"][i])
            scl = float(normalizer["scale"][i])
            input_expr = f"({input_names[i]} - ({off:.12g}))/({scl:.12g})"
        else:
            off = np.nan
            scl = np.nan
            input_expr = f"{input_names[i]}(x)"

        rows.append({
            "input_index": i,
            "input_name": input_names[i],
            "input_type": input_types[i],
            "activity_sd": float(activity_sd[i]),
            "mean_abs_edge_output": float(mean_abs[i]),
            "mean_signed_edge_output": float(mean_signed[i]),
            "mask": float(layer.mask[i, TARGET_NODE].detach().cpu()),
            "scale_base": float(layer.scale_base[i, TARGET_NODE].detach().cpu()),
            "scale_sp": float(layer.scale_sp[i, TARGET_NODE].detach().cpu()),
            "k": k,
            "raw_normalizer_offset": off,
            "raw_normalizer_scale": scl,
            "input_expression": input_expr,
            "coefficients_json": arr_json(coef),
            "grid_json": arr_json(grid),
        })

    ranking = pd.DataFrame(rows).sort_values("activity_sd", ascending=False).reset_index(drop=True)
    ranking.insert(0, "activity_rank", np.arange(1, len(ranking) + 1))
    ranking.to_csv(OUTDIR / "node17_incoming_edges_ranked.csv", index=False)

    source_summary = (
        ranking.groupby("input_type")
        .agg(
            n_edges=("input_index", "count"),
            sum_activity_sd=("activity_sd", "sum"),
            mean_activity_sd=("activity_sd", "mean"),
            max_activity_sd=("activity_sd", "max"),
            sum_mean_abs=("mean_abs_edge_output", "sum"),
        )
        .reset_index()
    )
    source_summary.to_csv(OUTDIR / "node17_source_type_summary.csv", index=False)

    sbn_scale = float(cs_kan.subnode_scale[0][TARGET_NODE].detach().cpu())
    sbn_bias = float(cs_kan.subnode_bias[0][TARGET_NODE].detach().cpu())
    nd_scale = float(cs_kan.node_scale[0][TARGET_NODE].detach().cpu())
    nd_bias = float(cs_kan.node_bias[0][TARGET_NODE].detach().cpu())
    base_fun_name = getattr(cs_kan, "base_fun_name", "silu")

    lines = [
        "EXACT CAUSE-SPECIFIC KAN NODE EQUATION",
        "=" * 100,
        f"Outer = {OUTER}",
        f"Target additive node = {TARGET_NODE}",
        f"Original features = {num_original}",
        f"Shared outputs = {num_shared}",
        f"Total CS inputs = {num_cs_inputs}",
        f"CS additive nodes = {dim_sum}",
        f"CS multiplication nodes = {dim_mult}",
        f"B-spline degree k = {k}",
        f"Base function = {base_fun_name}",
        "",
        f"r_{TARGET_NODE}(q) = SUM_i phi_i,{TARGET_NODE}(q_i)",
        f"subnode_{TARGET_NODE}(q) = ({sbn_scale:.12g})*r_{TARGET_NODE}(q) + ({sbn_bias:.12g})",
        f"c_{TARGET_NODE}(q) = ({nd_scale:.12g})*subnode_{TARGET_NODE}(q) + ({nd_bias:.12g})",
        "",
        f"phi_i,{TARGET_NODE}(v) = mask_i * [scale_base_i*{base_fun_name}(v) + scale_sp_i*SUM_r c_i,r B_(r,{k})(v)]",
        "",
        "FULL NUMERICAL PARAMETERIZATION",
        "=" * 100,
    ]

    for _, row in ranking.iterrows():
        i = int(row["input_index"])
        coef = json.loads(row["coefficients_json"])
        coef_expr = " ".join(
            f"{fmt(c)}*B_{r},{k}(v_{i})"
            for r, c in enumerate(coef)
        )
        lines.extend([
            "",
            f"[rank {int(row['activity_rank']):03d}] input {i:03d}: {row['input_name']} ({row['input_type']})",
            f"activity_SD = {row['activity_sd']:.12g}",
            f"v_{i} = {row['input_expression']}",
            f"phi_{i},{TARGET_NODE}(v_{i}) = ({row['mask']:.12g}) * [({row['scale_base']:.12g})*{base_fun_name}(v_{i}) + ({row['scale_sp']:.12g})*S_{i}(v_{i})]",
            f"S_{i}(v_{i}) = {coef_expr}",
        ])

    lines.extend([
        "",
        "=" * 100,
        "EXACT RECONSTRUCTION CHECKS",
        f"max edge error = {float(edge_err.max()):.12e}",
        f"mean edge error = {float(edge_err.mean()):.12e}",
        f"max layer-node error = {float(layer_node_err.max()):.12e}",
        f"mean layer-node error = {float(layer_node_err.mean()):.12e}",
        f"max final-node error = {float(final_node_err.max()):.12e}",
        f"mean final-node error = {float(final_node_err.mean()):.12e}",
    ])

    (OUTDIR / "exact_node17_equation.txt").write_text("\n".join(lines), encoding="utf-8")

    print("\n" + "=" * 100)
    print("EXACT CS-KAN NODE RECONSTRUCTION")
    print("=" * 100)
    print(f"Outer: {OUTER}")
    print(f"Target node: {TARGET_NODE}")
    print(f"Original features: {num_original}")
    print(f"Shared outputs: {num_shared}")
    print(f"Total CS inputs: {num_cs_inputs}")
    print(f"CS structure: {dim_sum} additive + {dim_mult} multiplicative nodes")
    print(f"k = {k}")

    print("\nTOP 15 INCOMING EDGES BY ACTIVITY SD")
    cols = [
        "activity_rank", "input_index", "input_name",
        "input_type", "activity_sd", "mean_abs_edge_output", "mask"
    ]
    print(ranking.loc[:, cols].head(15).to_string(index=False))

    age_row = ranking[ranking["input_name"] == "age_at_diagnosis"]
    if len(age_row) == 1:
        r = age_row.iloc[0]
        print("\nAGE WITHIN THIS NODE")
        print(
            f"rank={int(r['activity_rank'])}/{num_cs_inputs} | "
            f"activity_SD={r['activity_sd']:.10f} | "
            f"mean_abs={r['mean_abs_edge_output']:.10f}"
        )

    print("\nSOURCE-TYPE SUMMARY")
    print(source_summary.to_string(index=False))

    print("\nEXACT RECONSTRUCTION CHECK")
    print(f"max incoming-edge error = {float(edge_err.max()):.12e}")
    print(f"mean incoming-edge error = {float(edge_err.mean()):.12e}")
    print(f"max summed-layer-node error = {float(layer_node_err.max()):.12e}")
    print(f"mean summed-layer-node error = {float(layer_node_err.mean()):.12e}")
    print(f"max FINAL node error = {float(final_node_err.max()):.12e}")
    print(f"mean FINAL node error = {float(final_node_err.mean()):.12e}")

    print("\nFILES")
    print(OUTDIR / "exact_node17_equation.txt")
    print(OUTDIR / "node17_incoming_edges_ranked.csv")
    print(OUTDIR / "node17_source_type_summary.csv")
    print("=" * 100)


if __name__ == "__main__":
    main()
