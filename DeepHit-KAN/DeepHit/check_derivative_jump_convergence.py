import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import import_data as impt
from class_DeepHit import Model_DeepHit
from get_main_CTD_1L import get_network_settings


# ============================================================
# CONFIG
# ============================================================
OUTERS = [2, 4]  # Outer 2 -> k=3 ; Outer 4 -> k=1
FEATURE_NAME = "age_at_diagnosis"

BASE = Path("METABRIC/results_KAN1L_CTD")
FEATURE_FILE = Path("./sample data/METABRIC/cleaned_features_final.csv")
OUTDIR = Path("KAN1L_age_derivative_convergence")

# Fractions of the local knot spacing used as epsilon.
# We progressively approach each knot from left and right.
EPS_FRACTIONS = [0.10, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]


def load_normalizer(path):
    nf = np.load(path)
    return {
        "mode": str(nf["mode"]),
        "offset": nf["offset"],
        "scale": nf["scale"],
    }


def edge_value_and_d1(layer, input_index, output_index, x_raw, offset, scale):
    """
    Evaluate one real learned KAN edge and its first derivative dphi/dx
    with respect to the ORIGINAL feature scale.
    """
    x_norm_value = (float(x_raw) - offset) / scale
    x_norm = torch.tensor(
        x_norm_value,
        dtype=torch.float32,
        requires_grad=True
    )

    # Only this input coordinate matters for this edge activation.
    parts = []
    for d in range(layer.in_dim):
        if d == input_index:
            parts.append(x_norm.reshape(1))
        else:
            parts.append(torch.zeros(1, dtype=torch.float32))
    H = torch.stack(parts, dim=1)

    _, _, postacts, _ = layer(H)
    y = postacts[0, output_index, input_index]

    d1_norm = torch.autograd.grad(
        y,
        x_norm,
        create_graph=False,
        retain_graph=False
    )[0]

    # Chain rule: x_norm = (x_raw - offset) / scale
    d1_raw = float(d1_norm.detach()) / scale

    return float(y.detach()), d1_raw


def get_model_and_edge(out_itr):
    """
    Recreate the exact winning checkpoint and choose the strongest DIRECT
    age -> cause-specific edge using OUTER DEVELOPMENT ONLY.
    """
    df = pd.read_csv(FEATURE_FILE)
    feature_names = list(df.columns)
    if FEATURE_NAME not in feature_names:
        raise RuntimeError(f"{FEATURE_NAME} not found in feature CSV.")

    feature_idx = feature_names.index(FEATURE_NAME)
    raw_data = df.to_numpy(dtype=float)

    x_dim, DATA, MASK = impt.import_dataset_METABRIC()
    mask1, _ = MASK

    all_idx = np.arange(len(raw_data))
    dev_idx, test_idx = train_test_split(
        all_idx,
        test_size=0.20,
        random_state=1234 + out_itr,
    )

    itr_dir = BASE / f"itr_{out_itr}"
    hp_path = itr_dir / "hyperparameters_log.json"
    model_path = itr_dir / "models" / f"model_itr_{out_itr}.pt"
    norm_path = itr_dir / "models" / f"normalizer_itr_{out_itr}.npz"

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
        get_network_settings(hp, seed=1234 + out_itr),
    )
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    Xn = impt.apply_normalizer(raw_data, normalizer)
    Xdev = torch.tensor(Xn[dev_idx], dtype=torch.float32)

    with torch.no_grad():
        shared_dev = model.shared_kan(Xdev)
        h_cs_dev = torch.cat([Xdev, shared_dev], dim=1)

    cs_layer = model.cs_kans[0].act_fun[0]

    with torch.no_grad():
        _, _, postacts_dev, _ = cs_layer(h_cs_dev)

    edge_std_age = torch.std(
        postacts_dev[:, :, feature_idx],
        dim=0,
        unbiased=False
    ).cpu().numpy()

    hidden_node = int(np.argmax(edge_std_age))
    edge_activity = float(edge_std_age[hidden_node])

    offset = float(normalizer["offset"][feature_idx])
    scale = float(normalizer["scale"][feature_idx])

    k = int(cs_layer.k)
    grid_norm = cs_layer.grid[feature_idx].detach().cpu().numpy()
    core_grid_norm = grid_norm[k:-k] if k > 0 else grid_norm.copy()
    core_grid_raw = np.sort(core_grid_norm * scale + offset)

    age_dev = raw_data[dev_idx, feature_idx]
    age_dev = age_dev[np.isfinite(age_dev)]
    lo = float(np.quantile(age_dev, 0.01))
    hi = float(np.quantile(age_dev, 0.99))

    return {
        "model": model,
        "layer": cs_layer,
        "feature_idx": feature_idx,
        "hidden_node": hidden_node,
        "edge_activity": edge_activity,
        "offset": offset,
        "scale": scale,
        "k": k,
        "knots": core_grid_raw,
        "lo": lo,
        "hi": hi,
        "n_dev": len(dev_idx),
        "n_test": len(test_idx),
    }


def analyze_outer(out_itr):
    info = get_model_and_edge(out_itr)

    layer = info["layer"]
    feature_idx = info["feature_idx"]
    hidden_node = info["hidden_node"]
    offset = info["offset"]
    scale = info["scale"]
    k = info["k"]
    knots = info["knots"]
    lo = info["lo"]
    hi = info["hi"]

    outdir = OUTDIR / f"outer_{out_itr}_k{k}"
    outdir.mkdir(parents=True, exist_ok=True)

    # Unique knots only.
    unique_knots = np.unique(knots)

    rows = []

    # Interior knots only, and only inside the development-supported range.
    for pos in range(1, len(unique_knots) - 1):
        knot = float(unique_knots[pos])

        if not (lo < knot < hi):
            continue

        left_gap = knot - float(unique_knots[pos - 1])
        right_gap = float(unique_knots[pos + 1]) - knot
        local_gap = min(left_gap, right_gap)

        if local_gap <= 0:
            continue

        for frac in EPS_FRACTIONS:
            eps = frac * local_gap

            phi_left, d1_left = edge_value_and_d1(
                layer, feature_idx, hidden_node,
                knot - eps, offset, scale
            )
            phi_right, d1_right = edge_value_and_d1(
                layer, feature_idx, hidden_node,
                knot + eps, offset, scale
            )

            rows.append({
                "outer": out_itr,
                "k": k,
                "hidden_node": hidden_node,
                "knot_age": knot,
                "local_gap_years": local_gap,
                "eps_fraction": frac,
                "epsilon_years": eps,
                "phi_left": phi_left,
                "phi_right": phi_right,
                "phi_gap": phi_right - phi_left,
                "abs_phi_gap": abs(phi_right - phi_left),
                "d1_left": d1_left,
                "d1_right": d1_right,
                "d1_jump": d1_right - d1_left,
                "abs_d1_jump": abs(d1_right - d1_left),
            })

    df = pd.DataFrame(rows)

    if df.empty:
        raise RuntimeError(
            f"No interior knots found inside the development range for outer {out_itr}."
        )

    df.to_csv(outdir / "all_knots_epsilon_convergence.csv", index=False)

    # Find the knot with the largest jump at the same reference fraction (0.02)
    # used in the previous script.
    ref = df[np.isclose(df["eps_fraction"], 0.02)].copy()
    idx = ref["abs_d1_jump"].idxmax()
    representative_knot = float(ref.loc[idx, "knot_age"])

    focus = df[np.isclose(df["knot_age"], representative_knot)].copy()
    focus = focus.sort_values("eps_fraction", ascending=False)
    focus.to_csv(outdir / "max_jump_knot_convergence.csv", index=False)

    # Also summarize the maximum derivative jump across ALL interior knots
    # for each epsilon fraction.
    summary = (
        df.groupby("eps_fraction", as_index=False)
          .agg(
              max_abs_d1_jump=("abs_d1_jump", "max"),
              median_abs_d1_jump=("abs_d1_jump", "median"),
              max_abs_phi_gap=("abs_phi_gap", "max"),
          )
          .sort_values("eps_fraction", ascending=False)
    )
    summary.to_csv(outdir / "global_convergence_summary.csv", index=False)

    # Plot: selected knot
    fig = plt.figure(figsize=(7.5, 4.8))
    ax = fig.add_subplot(111)
    ax.plot(
        focus["epsilon_years"],
        focus["abs_d1_jump"],
        marker="o"
    )
    ax.set_xscale("log")
    ax.set_xlabel(r"$\epsilon$ (years, log scale)")
    ax.set_ylabel(r"$|\phi'(x_k+\epsilon)-\phi'(x_k-\epsilon)|$")
    ax.set_title(
        f"Outer {out_itr}, k={k}: derivative jump convergence\n"
        f"age knot = {representative_knot:.3f} years"
    )
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(
        outdir / "max_jump_knot_convergence.png",
        dpi=220,
        bbox_inches="tight"
    )
    plt.close(fig)

    # Plot: maximum jump across all knots
    fig = plt.figure(figsize=(7.5, 4.8))
    ax = fig.add_subplot(111)
    ax.plot(
        summary["eps_fraction"],
        summary["max_abs_d1_jump"],
        marker="o"
    )
    ax.set_xscale("log")
    ax.set_xlabel("epsilon / local knot spacing (log scale)")
    ax.set_ylabel("Maximum |D1 jump| across interior knots")
    ax.set_title(
        f"Outer {out_itr}, k={k}: global derivative-jump convergence"
    )
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(
        outdir / "global_jump_convergence.png",
        dpi=220,
        bbox_inches="tight"
    )
    plt.close(fig)

    print("\n" + "=" * 90)
    print(f"OUTER {out_itr} | k={k}")
    print("=" * 90)
    print(f"Development patients: {info['n_dev']}")
    print(f"Outer test patients excluded from interpretation selection: {info['n_test']}")
    print(f"Strongest DIRECT age -> CS node: {hidden_node}")
    print(f"Edge activity SD (development): {info['edge_activity']:.6f}")
    print(f"Representative knot = {representative_knot:.6f} years")
    print("\nCONVERGENCE AT REPRESENTATIVE KNOT")
    print(
        focus[
            [
                "eps_fraction",
                "epsilon_years",
                "phi_gap",
                "d1_left",
                "d1_right",
                "d1_jump",
                "abs_d1_jump",
            ]
        ].to_string(index=False)
    )

    print("\nGLOBAL MAXIMUM |D1 JUMP| ACROSS ALL INTERIOR KNOTS")
    print(summary.to_string(index=False))

    smallest = focus.sort_values("eps_fraction").iloc[0]
    print("\nSMALLEST EPSILON RESULT")
    print(
        f"eps_fraction = {smallest['eps_fraction']:.6f} | "
        f"epsilon = {smallest['epsilon_years']:.6e} years | "
        f"|D1 jump| = {smallest['abs_d1_jump']:.6e}"
    )

    print(f"\nSaved results in: {outdir}")


def main():
    OUTDIR.mkdir(exist_ok=True)

    for out_itr in OUTERS:
        analyze_outer(out_itr)

    print("\n" + "=" * 90)
    print("DONE")
    print("Output folder:", OUTDIR.resolve())
    print("=" * 90)


if __name__ == "__main__":
    main()
