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
BASE = Path("METABRIC/results_KAN1L_CTD")
FEATURE_FILE = Path("./sample data/METABRIC/cleaned_features_final.csv")
OUTDIR = Path("KAN1L_exact_formulas_outer2_key_paths")

KEY_PATHS = [
    # Direct dominant paths
    ("CS", "age_at_diagnosis", None, 27, "age -> c27"),
    ("CS", "NPI", None, 9, "NPI -> c9"),

    # Dominant shared branch
    ("SHARED", "age_at_diagnosis", None, 41, "age -> s41"),
    ("SHARED", "NPI", None, 41, "NPI -> s41"),
    ("CS_SHARED", None, 41, 19, "s41 -> c19"),
]


def load_normalizer(path):
    nf = np.load(path)
    return {
        "mode": str(nf["mode"]),
        "offset": nf["offset"],
        "scale": nf["scale"],
    }


def edge_record(layer, in_idx, out_idx, input_label):
    k = int(layer.k)
    mask = float(layer.mask[in_idx, out_idx].detach())
    scale_base = float(layer.scale_base[in_idx, out_idx].detach())
    scale_sp = float(layer.scale_sp[in_idx, out_idx].detach())
    coef = layer.coef[in_idx, out_idx, :].detach().cpu().numpy()
    full_grid = layer.grid[in_idx, :].detach().cpu().numpy()
    core_grid = full_grid[k:-k] if k > 0 else full_grid.copy()

    return {
        "input_label": input_label,
        "input_index": int(in_idx),
        "output_index": int(out_idx),
        "k": k,
        "mask": mask,
        "scale_base": scale_base,
        "scale_sp": scale_sp,
        "coef": coef,
        "core_grid": core_grid,
    }


def node_affine(model_kan, node_idx):
    sub_scale = float(model_kan.subnode_scale[0][node_idx].detach())
    sub_bias = float(model_kan.subnode_bias[0][node_idx].detach())
    node_scale = float(model_kan.node_scale[0][node_idx].detach())
    node_bias = float(model_kan.node_bias[0][node_idx].detach())

    A = node_scale * sub_scale
    B = node_scale * sub_bias + node_bias

    return {
        "subnode_scale": sub_scale,
        "subnode_bias": sub_bias,
        "node_scale": node_scale,
        "node_bias": node_bias,
        "collapsed_A": A,
        "collapsed_B": B,
    }


def print_edge(title, rec, raw_offset=None, raw_scale=None, raw_values=None):
    print("\n" + "=" * 110)
    print(title)
    print("=" * 110)

    print(f"input_label = {rec['input_label']}")
    print(f"input_index = {rec['input_index']}")
    print(f"output_index = {rec['output_index']}")
    print(f"k = {rec['k']}")
    print(f"mask = {rec['mask']:.10f}")
    print(f"scale_base = {rec['scale_base']:.10f}")
    print(f"scale_sp = {rec['scale_sp']:.10f}")
    print("base_fun = SiLU")

    print("\nEXACT EDGE FORM")
    print(
        f"phi(v) = {rec['mask']:.10f} * ["
        f"{rec['scale_base']:.10f} * SiLU(v) + "
        f"{rec['scale_sp']:.10f} * SUM_r c_r B_(r,{rec['k']})(v)]"
    )

    if raw_offset is not None and raw_scale is not None:
        print(
            f"where v = (x - ({float(raw_offset):.10f})) "
            f"/ ({float(raw_scale):.10f})"
        )

    print("\nB-SPLINE COEFFICIENTS")
    for r, c in enumerate(rec["coef"]):
        print(f"c[{r:02d}] = {c:+.10f}")

    print("\nCORE KNOTS — MODEL INPUT SCALE")
    print(np.array2string(rec["core_grid"], precision=8, separator=", "))

    if raw_offset is not None and raw_scale is not None:
        raw_knots = rec["core_grid"] * float(raw_scale) + float(raw_offset)
        print("\nCORE KNOTS — PRE-NORMALIZATION SCALE")
        print(np.array2string(raw_knots, precision=8, separator=", "))

    if raw_values is not None:
        uniq = np.unique(raw_values)
        print(f"\nObserved unique raw values = {len(uniq)}")
        if len(uniq) <= 20:
            print(np.array2string(uniq, precision=8, separator=", "))
        else:
            print(
                f"Observed raw range = "
                f"[{float(np.min(raw_values)):.6f}, {float(np.max(raw_values)):.6f}]"
            )


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
        shared_out = model.shared_kan(Xdev)
        q = torch.cat([Xdev, shared_out], dim=1)

    shared = model.shared_kan
    cs = model.cs_kans[0]

    if len(shared.act_fun) != 1 or len(cs.act_fun) != 1:
        raise RuntimeError("This script is for 1-layer Outer-2 KANs.")

    sh_layer = shared.act_fun[0]
    cs_layer = cs.act_fun[0]

    n_original = Xdev.shape[1]

    print("\n" + "#" * 110)
    print("EXACT FORMULAS FOR DOMINANT AGE / NPI PATHS — OUTER 2")
    print("#" * 110)

    print("\nGENERAL EXACT HIERARCHICAL FORM FOR THIS ADDITIVE OUTER-2 MODEL")
    print("For each Shared node k:")
    print("    s_k(x) = A^S_k * SUM_i phi^S_(i,k)(u_i) + B^S_k")
    print("For each CS node j:")
    print("    c_j(x) = A^C_j * [SUM_i phi^C_(i,j)(u_i)")
    print("                         + SUM_k psi^C_(k,j)(s_k(x))] + B^C_j")
    print("Final logits:")
    print("    z_t(x) = b_t + SUM_j W_(t,j) c_j(x)")
    print("Final CIF:")
    print("    F_h(x) = SUM_(t<=h) exp(z_t(x)) / SUM_s exp(z_s(x))")

    # ----------------------------------------------------------
    # Selected node affine transforms
    # ----------------------------------------------------------
    selected_shared_nodes = [41]
    selected_cs_nodes = [9, 19, 27]

    print("\nSELECTED NODE AFFINE TRANSFORMS")
    rows_affine = []

    for k in selected_shared_nodes:
        a = node_affine(shared, k)
        rows_affine.append({
            "node": f"s{k}",
            **a,
        })

    for j in selected_cs_nodes:
        a = node_affine(cs, j)
        rows_affine.append({
            "node": f"c{j}",
            **a,
        })

    affine_df = pd.DataFrame(rows_affine)
    print(affine_df.to_string(index=False))
    affine_df.to_csv(
        OUTDIR / "selected_node_affines.csv",
        index=False,
    )

    # ----------------------------------------------------------
    # Print/save exact edges
    # ----------------------------------------------------------
    all_param_rows = []

    for kind, feature_name, shared_idx, out_idx, label in KEY_PATHS:
        if kind == "SHARED":
            in_idx = feature_names.index(feature_name)
            rec = edge_record(
                sh_layer,
                in_idx,
                out_idx,
                f"u_{feature_name}",
            )
            raw_offset = normalizer["offset"][in_idx]
            raw_scale = normalizer["scale"][in_idx]
            raw_values = raw_data[dev_idx, in_idx]

        elif kind == "CS":
            in_idx = feature_names.index(feature_name)
            rec = edge_record(
                cs_layer,
                in_idx,
                out_idx,
                f"u_{feature_name}",
            )
            raw_offset = normalizer["offset"][in_idx]
            raw_scale = normalizer["scale"][in_idx]
            raw_values = raw_data[dev_idx, in_idx]

        elif kind == "CS_SHARED":
            in_idx = n_original + shared_idx
            rec = edge_record(
                cs_layer,
                in_idx,
                out_idx,
                f"s_{shared_idx}",
            )
            raw_offset = None
            raw_scale = None
            raw_values = None

        else:
            raise ValueError(kind)

        print_edge(
            label,
            rec,
            raw_offset=raw_offset,
            raw_scale=raw_scale,
            raw_values=raw_values,
        )

        for r, coef in enumerate(rec["coef"]):
            all_param_rows.append({
                "path_label": label,
                "kind": kind,
                "input_label": rec["input_label"],
                "input_index": rec["input_index"],
                "output_index": rec["output_index"],
                "k": rec["k"],
                "mask": rec["mask"],
                "scale_base": rec["scale_base"],
                "scale_sp": rec["scale_sp"],
                "coef_index": r,
                "coef_value": float(coef),
            })

    pd.DataFrame(all_param_rows).to_csv(
        OUTDIR / "key_edge_parameters.csv",
        index=False,
    )

    # ----------------------------------------------------------
    # Save compact exact formula text
    # ----------------------------------------------------------
    txt = []
    txt.append("EXACT HIERARCHICAL FORM — OUTER 2\n")
    txt.append(
        "s_k(x) = A^S_k * SUM_i phi^S_(i,k)(u_i) + B^S_k\n"
    )
    txt.append(
        "c_j(x) = A^C_j * [SUM_i phi^C_(i,j)(u_i) "
        "+ SUM_k psi^C_(k,j)(s_k(x))] + B^C_j\n"
    )
    txt.append(
        "z_t(x) = b_t + SUM_j W_(t,j)c_j(x)\n"
    )
    txt.append(
        "F_h(x) = SUM_(t<=h) exp(z_t(x)) / SUM_s exp(z_s(x))\n\n"
    )
    txt.append("Selected dominant routes:\n")
    txt.append("  age -> c27 -> F_h\n")
    txt.append("  age -> s41 -> c19 -> F_h\n")
    txt.append("  NPI -> c9 -> F_h\n")
    txt.append("  NPI -> s41 -> c19 -> F_h\n")

    (OUTDIR / "exact_hierarchical_formula.txt").write_text(
        "".join(txt),
        encoding="utf-8",
    )

    print("\nFILES")
    print(OUTDIR / "selected_node_affines.csv")
    print(OUTDIR / "key_edge_parameters.csv")
    print(OUTDIR / "exact_hierarchical_formula.txt")
    print("#" * 110)


if __name__ == "__main__":
    main()
