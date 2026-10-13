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


OUTERS = [2, 4]
FEATURE_NAME = "age_at_diagnosis"

BASE = Path("METABRIC/results_KAN1L_CTD")
FEATURE_FILE = Path("./sample data/METABRIC/cleaned_features_final.csv")
OUTDIR = Path("KAN1L_age_continuity_devonly")


def load_normalizer(path):
    nf = np.load(path)
    return {
        "mode": str(nf["mode"]),
        "offset": nf["offset"],
        "scale": nf["scale"],
    }


def eval_edge(layer, input_index, output_index, x_raw, offset, scale):
    x_raw = np.asarray(x_raw, dtype=float)
    x_norm = (x_raw - offset) / scale

    H = torch.zeros((len(x_norm), layer.in_dim), dtype=torch.float32)
    H[:, input_index] = torch.tensor(x_norm, dtype=torch.float32)

    with torch.no_grad():
        _, _, postacts, _ = layer(H)

    return postacts[:, output_index, input_index].cpu().numpy()


def edge_derivatives_autograd(layer, input_index, output_index, x_raw, offset, scale):
    x_norm_value = (float(x_raw) - offset) / scale
    x_norm = torch.tensor(x_norm_value, dtype=torch.float32, requires_grad=True)

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
        y, x_norm, create_graph=True, retain_graph=True
    )[0]

    try:
        d2_norm = torch.autograd.grad(
            d1_norm, x_norm, retain_graph=False
        )[0]
        d2_raw = float(d2_norm.detach()) / (scale ** 2)
    except RuntimeError:
        d2_raw = np.nan

    d1_raw = float(d1_norm.detach()) / scale
    return float(y.detach()), d1_raw, d2_raw


def analyze_outer(out_itr):
    print("\n" + "=" * 84)
    print(f"OUTER {out_itr}")
    print("=" * 84)

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
        unbiased=False,
    ).cpu().numpy()

    hidden_node = int(np.argmax(edge_std_age))
    strongest_std = float(edge_std_age[hidden_node])

    k = int(cs_layer.k)
    grid_norm = cs_layer.grid[feature_idx].detach().cpu().numpy()
    core_grid_norm = grid_norm[k:-k] if k > 0 else grid_norm.copy()

    offset = float(normalizer["offset"][feature_idx])
    scale = float(normalizer["scale"][feature_idx])
    core_grid_raw = core_grid_norm * scale + offset

    sorted_grid = np.sort(core_grid_raw)
    diffs = np.diff(sorted_grid)
    tol = max(1e-7, 1e-6 * max(1.0, float(np.ptp(sorted_grid))))
    near_duplicate = diffs <= tol
    n_near_duplicates = int(np.sum(near_duplicate))
    min_spacing = float(np.min(diffs)) if len(diffs) else np.nan

    age_dev = raw_data[dev_idx, feature_idx]
    age_dev = age_dev[np.isfinite(age_dev)]
    lo = float(np.quantile(age_dev, 0.01))
    hi = float(np.quantile(age_dev, 0.99))

    x_raw = np.linspace(lo, hi, 1200)
    phi = eval_edge(
        cs_layer,
        feature_idx,
        hidden_node,
        x_raw,
        offset,
        scale,
    )

    d1_plot = np.gradient(phi, x_raw)
    d2_plot = np.gradient(d1_plot, x_raw)

    outdir = OUTDIR / f"outer_{out_itr}_k{k}"
    outdir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({
        "age_at_diagnosis": x_raw,
        "phi": phi,
        "dphi_dage_numeric": d1_plot,
        "d2phi_dage2_numeric": d2_plot,
    }).to_csv(outdir / "age_edge_curve.csv", index=False)

    knot_rows = []
    unique_knots = np.unique(sorted_grid)

    for pos in range(1, len(unique_knots) - 1):
        knot = float(unique_knots[pos])

        if not (lo < knot < hi):
            continue

        left_gap = knot - float(unique_knots[pos - 1])
        right_gap = float(unique_knots[pos + 1]) - knot
        local_gap = min(left_gap, right_gap)

        if local_gap <= tol:
            continue

        eps = 0.02 * local_gap

        y_left, d1_left, d2_left = edge_derivatives_autograd(
            cs_layer, feature_idx, hidden_node,
            knot - eps, offset, scale
        )
        y_right, d1_right, d2_right = edge_derivatives_autograd(
            cs_layer, feature_idx, hidden_node,
            knot + eps, offset, scale
        )

        knot_rows.append({
            "knot_age": knot,
            "epsilon_years": eps,
            "phi_left": y_left,
            "phi_right": y_right,
            "phi_gap_at_pm_epsilon": y_right - y_left,
            "d1_left": d1_left,
            "d1_right": d1_right,
            "d1_jump_approx": d1_right - d1_left,
            "abs_d1_jump_approx": abs(d1_right - d1_left),
            "d2_left": d2_left,
            "d2_right": d2_right,
            "d2_jump_approx": d2_right - d2_left
                if np.isfinite(d2_left) and np.isfinite(d2_right)
                else np.nan,
        })

    knots_df = pd.DataFrame(knot_rows)
    knots_df.to_csv(outdir / "age_knot_derivative_checks.csv", index=False)

    for values, ylabel, filename, title_part in [
        (phi, r"$\phi(x)$", "age_phi.png", "learned edge function"),
        (d1_plot, r"$d\phi/dx$", "age_d1.png", "first derivative"),
        (d2_plot, r"$d^2\phi/dx^2$", "age_d2.png", "second derivative"),
    ]:
        fig = plt.figure(figsize=(8, 4.8))
        ax = fig.add_subplot(111)
        ax.plot(x_raw, values)
        for knot in unique_knots:
            if lo < knot < hi:
                ax.axvline(knot, alpha=0.15)
        ax.set_xlabel("Age at diagnosis (years)")
        ax.set_ylabel(ylabel)
        ax.set_title(
            f"Outer {out_itr}, k={k}: age -> CS node {hidden_node} ({title_part})"
        )
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(outdir / filename, dpi=220, bbox_inches="tight")
        plt.close(fig)

    print(f"k = {k}")
    print(f"Development patients used for interpretation selection: {len(dev_idx)}")
    print(f"Outer test patients kept out of selection: {len(test_idx)}")
    print(f"Strongest DIRECT age -> CS node: {hidden_node}")
    print(f"Edge activity SD on development set: {strongest_std:.6f}")
    print(f"Age DEV 1%-99% range: {lo:.3f} to {hi:.3f} years")
    print("Core learned knots in ORIGINAL age scale:")
    print(np.array2string(sorted_grid, precision=3, separator=", "))
    print(f"Minimum knot spacing: {min_spacing:.6e}")
    print(f"Near-duplicate knot spacings (tol={tol:.3e}): {n_near_duplicates}")

    theoretical_continuity = f"C^{k-1}" if n_near_duplicates == 0 else (
        "cannot claim C^(k-1) globally without accounting for repeated/near-repeated knots"
    )
    print(f"Theoretical spline continuity check: {theoretical_continuity}")

    if len(knots_df):
        cols = [
            "knot_age",
            "d1_left",
            "d1_right",
            "d1_jump_approx",
            "d2_left",
            "d2_right",
            "d2_jump_approx",
        ]
        print("\nONE-SIDED DERIVATIVE CHECKS AROUND INTERIOR KNOTS")
        print(knots_df[cols].to_string(index=False))
        print(
            "\nMax approximate |D1 jump|: "
            f"{knots_df['abs_d1_jump_approx'].max():.6e}"
        )
    else:
        print("\nNo interior knots inside the DEV-supported 1%-99% age range.")

    print(f"\nSaved results in: {outdir}")


def main():
    OUTDIR.mkdir(exist_ok=True)

    for out_itr in OUTERS:
        analyze_outer(out_itr)

    print("\n" + "=" * 84)
    print("DONE")
    print("Output folder:", OUTDIR.resolve())
    print("=" * 84)


if __name__ == "__main__":
    main()
