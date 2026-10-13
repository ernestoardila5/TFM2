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
FEATURE_NAME = "age_at_diagnosis"

BASE = Path("METABRIC/results_KAN1L_CTD")
FEATURE_FILE = Path("./sample data/METABRIC/cleaned_features_final.csv")
OUTDIR = Path("KAN1L_exact_formula_outer2_age")


def load_normalizer(path):
    nf = np.load(path)
    return {
        "mode": str(nf["mode"]),
        "offset": nf["offset"],
        "scale": nf["scale"],
    }


def fmt(x):
    return f"{float(x):+.10g}"


def main():
    OUTDIR.mkdir(exist_ok=True)

    df = pd.read_csv(FEATURE_FILE)
    feature_names = list(df.columns)
    raw_data = df.to_numpy(dtype=float)

    if FEATURE_NAME not in feature_names:
        raise RuntimeError(f"{FEATURE_NAME} not found.")

    feature_idx = feature_names.index(FEATURE_NAME)

    x_dim, DATA, MASK = impt.import_dataset_METABRIC()
    mask1, _ = MASK

    all_idx = np.arange(len(raw_data))
    dev_idx, test_idx = train_test_split(
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

    Xn = impt.apply_normalizer(raw_data, normalizer)
    Xdev = torch.tensor(Xn[dev_idx], dtype=torch.float32)

    with torch.no_grad():
        shared_dev = model.shared_kan(Xdev)
        h_cs_dev = torch.cat([Xdev, shared_dev], dim=1)

    cs_kan = model.cs_kans[0]
    layer = cs_kan.act_fun[0]

    with torch.no_grad():
        _, _, postacts_dev, _ = layer(h_cs_dev)

    edge_std_age = torch.std(
        postacts_dev[:, :, feature_idx],
        dim=0,
        unbiased=False,
    ).cpu().numpy()

    node = int(np.argmax(edge_std_age))
    edge_activity = float(edge_std_age[node])

    k = int(layer.k)
    mask = float(layer.mask[feature_idx, node].detach().cpu())
    scale_base = float(layer.scale_base[feature_idx, node].detach().cpu())
    scale_sp = float(layer.scale_sp[feature_idx, node].detach().cpu())

    coef = (
        layer.coef[feature_idx, node, :]
        .detach()
        .cpu()
        .numpy()
        .astype(float)
    )

    grid_norm = (
        layer.grid[feature_idx]
        .detach()
        .cpu()
        .numpy()
        .astype(float)
    )

    age_offset = float(normalizer["offset"][feature_idx])
    age_scale = float(normalizer["scale"][feature_idx])

    grid_raw = grid_norm * age_scale + age_offset
    core_grid_norm = grid_norm[k:-k] if k > 0 else grid_norm.copy()
    core_grid_raw = core_grid_norm * age_scale + age_offset

    base_fun_name = getattr(cs_kan, "base_fun_name", None)
    if base_fun_name is None:
        base_fun_name = type(layer.base_fun).__name__

    age_raw_dev = raw_data[dev_idx, feature_idx]
    u_dev = (age_raw_dev - age_offset) / age_scale

    H = torch.zeros(
        (len(u_dev), layer.in_dim),
        dtype=torch.float32
    )
    H[:, feature_idx] = torch.tensor(u_dev, dtype=torch.float32)

    with torch.no_grad():
        _, _, postacts_check, _ = layer(H)

        basis_all = B_batch(H, layer.grid, k=k)
        basis_age = basis_all[:, feature_idx, :]

        coef_t = layer.coef[feature_idx, node, :]
        spline_part = basis_age @ coef_t

        base_part = layer.base_fun(H[:, feature_idx])

        manual = mask * (
            scale_base * base_part
            + scale_sp * spline_part
        )

        actual = postacts_check[:, node, feature_idx]

    abs_err = torch.abs(manual - actual)
    max_abs_err = float(abs_err.max().cpu())
    mean_abs_err = float(abs_err.mean().cpu())

    subnode_scale = np.nan
    subnode_bias = np.nan
    node_scale = np.nan
    node_bias = np.nan

    try:
        subnode_scale = float(
            cs_kan.subnode_scale[0][node].detach().cpu()
        )
        subnode_bias = float(
            cs_kan.subnode_bias[0][node].detach().cpu()
        )
    except Exception:
        pass

    try:
        node_scale = float(
            cs_kan.node_scale[0][node].detach().cpu()
        )
        node_bias = float(
            cs_kan.node_bias[0][node].detach().cpu()
        )
    except Exception:
        pass

    pd.DataFrame({
        "basis_index_r": np.arange(len(coef)),
        "coefficient_c_r": coef,
    }).to_csv(
        OUTDIR / "age_edge_bspline_coefficients.csv",
        index=False
    )

    pd.DataFrame({
        "grid_index": np.arange(len(grid_norm)),
        "grid_normalized": grid_norm,
        "grid_raw_age_years": grid_raw,
        "is_core_grid": [
            (k <= i < len(grid_norm) - k) if k > 0 else True
            for i in range(len(grid_norm))
        ],
    }).to_csv(
        OUTDIR / "age_edge_knot_vector.csv",
        index=False
    )

    coef_terms = " ".join(
        [f"{fmt(c)}*B_{r},{k}(u)" for r, c in enumerate(coef)]
    )

    formula_lines = [
        "EXACT LEARNED KAN EDGE FORMULA",
        "=" * 80,
        f"Outer: {OUTER}",
        f"Feature: {FEATURE_NAME}",
        f"Feature index: {feature_idx}",
        f"Cause-specific output/subnode index: {node}",
        f"k (B-spline degree): {k}",
        f"Base function: {base_fun_name}",
        f"Development edge-activity SD: {edge_activity:.10g}",
        "",
        "RAW-TO-NORMALIZED INPUT:",
        f"u = (age - ({age_offset:.12g})) / ({age_scale:.12g})",
        "",
        "EXACT EDGE FORM:",
        "phi_age,node(age) = mask * [scale_base * base_fun(u) + scale_sp * S_k(u)]",
        "",
        f"mask       = {mask:.12g}",
        f"scale_base = {scale_base:.12g}",
        f"scale_sp   = {scale_sp:.12g}",
        "",
        "where",
        f"S_{k}(u) = SUM_r c_r * B_(r,{k})(u)",
        "",
        "Expanded B-spline representation:",
        f"S_{k}(u) = {coef_terms}",
        "",
        "CORE KNOTS IN RAW AGE SCALE:",
        np.array2string(core_grid_raw, precision=6, separator=", "),
        "",
        "FULL EXTENDED KNOT VECTOR IN NORMALIZED SCALE:",
        np.array2string(grid_norm, precision=8, separator=", "),
        "",
        "FULL EXTENDED KNOT VECTOR IN RAW AGE SCALE:",
        np.array2string(grid_raw, precision=6, separator=", "),
        "",
        "EXACT RECONSTRUCTION CHECK:",
        f"max |manual - KANLayer.forward| = {max_abs_err:.12e}",
        f"mean |manual - KANLayer.forward| = {mean_abs_err:.12e}",
        "",
        "POST-SUM AFFINE PARAMETERS FOR THIS CS NODE:",
        f"subnode_scale = {subnode_scale}",
        f"subnode_bias  = {subnode_bias}",
        f"node_scale    = {node_scale}",
        f"node_bias     = {node_bias}",
        "",
        "IMPORTANT:",
        "This is the exact checkpoint representation in the learned B-spline basis.",
        "It is NOT yet a simplified symbolic approximation.",
    ]

    (OUTDIR / "exact_age_edge_formula.txt").write_text(
        "\n".join(formula_lines),
        encoding="utf-8"
    )

    print("\n" + "=" * 90)
    print("EXACT KAN EDGE EXTRACTION")
    print("=" * 90)
    print(f"Outer: {OUTER}")
    print(f"Feature: {FEATURE_NAME} (index {feature_idx})")
    print(f"Selected CS node/subnode: {node}")
    print(f"k = {k}")
    print(f"Base function = {base_fun_name}")
    print(f"Activity SD = {edge_activity:.8f}")

    print("\nRAW -> NORMALIZED")
    print(
        f"u = (age - ({age_offset:.10f})) / "
        f"({age_scale:.10f})"
    )

    print("\nEDGE PARAMETERS")
    print(f"mask       = {mask:.10f}")
    print(f"scale_base = {scale_base:.10f}")
    print(f"scale_sp   = {scale_sp:.10f}")

    print("\nB-SPLINE COEFFICIENTS c_r")
    for r, c in enumerate(coef):
        print(f"c[{r:02d}] = {c:+.10f}")

    print("\nCORE KNOTS (RAW AGE YEARS)")
    print(np.array2string(core_grid_raw, precision=5, separator=", "))

    print("\nEXACT RECONSTRUCTION CHECK")
    print(f"max absolute error  = {max_abs_err:.12e}")
    print(f"mean absolute error = {mean_abs_err:.12e}")

    print("\nNODE AFFINE PARAMETERS AFTER EDGE SUM")
    print(f"subnode_scale = {subnode_scale}")
    print(f"subnode_bias  = {subnode_bias}")
    print(f"node_scale    = {node_scale}")
    print(f"node_bias     = {node_bias}")

    print("\nFILES")
    print(OUTDIR / "exact_age_edge_formula.txt")
    print(OUTDIR / "age_edge_bspline_coefficients.csv")
    print(OUTDIR / "age_edge_knot_vector.csv")

    print("\nIf reconstruction error is ~1e-7 or smaller,")
    print("the extracted equation reproduces the actual trained KAN edge.")
    print("=" * 90)


if __name__ == "__main__":
    main()
