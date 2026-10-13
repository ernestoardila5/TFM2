import os
import re
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

import import_data as impt
from class_DeepHit import Model_DeepHit
from get_main_CTD_1L import get_network_settings


OUTERS = [2, 4]   # outer 2 => k=3 ; outer 4 => k=1
TOP_FEATURES = 5
BASE = Path("METABRIC/results_KAN1L_CTD")
FEATURE_FILE = Path("./sample data/METABRIC/cleaned_features_final.csv")
OUTPUT_ROOT = Path("KAN1L_function_analysis")


def sanitize(name):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name))[:120]


def load_normalizer(path):
    nf = np.load(path)
    return {
        "mode": str(nf["mode"]),
        "offset": nf["offset"],
        "scale": nf["scale"],
    }


def main_for_outer(out_itr):
    print("\n" + "=" * 78)
    print(f"OUTER {out_itr}")
    print("=" * 78)

    # Raw METABRIC feature names + values
    df = pd.read_csv(FEATURE_FILE)
    feature_names = list(df.columns)
    raw_data = df.to_numpy(dtype=float)

    # Dataset metadata/masks from the exact local pipeline
    x_dim, DATA, MASK = impt.import_dataset_METABRIC()
    _, time, label = DATA
    mask1, mask2 = MASK

    if raw_data.shape[1] != x_dim:
        raise RuntimeError(
            f"Feature CSV has {raw_data.shape[1]} columns, but model expects x_dim={x_dim}"
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
    Xn = impt.apply_normalizer(raw_data, normalizer)

    input_dims = {
        "x_dim": x_dim,
        "num_Event": mask1.shape[1],
        "num_Category": mask1.shape[2],
    }

    network_settings = get_network_settings(hp, seed=1234 + out_itr)
    model = Model_DeepHit(input_dims, network_settings)
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    print(f"k = {hp['k']}")
    print(f"grid = {hp['grid']}")
    print(f"shared: sum={hp['sum_nodes_shared']}, mult={hp['mult_nodes_shared']}")
    print(f"CS:     sum={hp['sum_nodes_CS']}, mult={hp['mult_nodes_CS']}")

    # First/shared KAN layer: edge functions phi_{i->j}(x_i)
    layer = model.shared_kan.act_fun[0]

    X_tensor = torch.tensor(Xn, dtype=torch.float32)
    with torch.no_grad():
        _, _, postacts, _ = layer(X_tensor)

    # postacts shape: [N, out_dim, in_dim]
    if postacts.ndim != 3:
        raise RuntimeError(f"Unexpected postacts shape: {tuple(postacts.shape)}")

    edge_std = torch.std(postacts, dim=0, unbiased=False).cpu().numpy()  # [out,in]

    # Aggregate edge activity per original input feature.
    # This is NOT a causal importance score; it measures how much learned
    # edge functions vary over the observed METABRIC values.
    feature_activity = np.sqrt(np.mean(edge_std ** 2, axis=0))

    rows = []
    for i, name in enumerate(feature_names):
        j = int(np.argmax(edge_std[:, i]))
        rows.append({
            "feature_index": i,
            "feature_name": name,
            "activity_rms_edge_std": float(feature_activity[i]),
            "strongest_hidden_node": j,
            "strongest_edge_std": float(edge_std[j, i]),
        })

    rank_df = pd.DataFrame(rows).sort_values(
        "activity_rms_edge_std", ascending=False
    ).reset_index(drop=True)

    outdir = OUTPUT_ROOT / f"outer_{out_itr}_k{hp['k']}"
    outdir.mkdir(parents=True, exist_ok=True)
    rank_df.to_csv(outdir / "feature_activity_ranking.csv", index=False)

    print("\nTOP 10 FEATURES BY SHARED-KAN EDGE ACTIVITY")
    print(rank_df.head(10).to_string(index=False))

    # Plot strongest learned edge for each of the top features.
    median_raw = np.nanmedian(raw_data, axis=0)

    for rank, row in rank_df.head(TOP_FEATURES).iterrows():
        i = int(row["feature_index"])
        j = int(row["strongest_hidden_node"])
        name = row["feature_name"]

        observed = raw_data[:, i]
        observed = observed[np.isfinite(observed)]

        lo = float(np.quantile(observed, 0.01))
        hi = float(np.quantile(observed, 0.99))

        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            print(f"Skipping {name}: invalid/constant range.")
            continue

        x_raw = np.linspace(lo, hi, 500)

        X_curve_raw = np.tile(median_raw, (len(x_raw), 1))
        X_curve_raw[:, i] = x_raw
        X_curve_norm = impt.apply_normalizer(X_curve_raw, normalizer)

        X_curve_tensor = torch.tensor(X_curve_norm, dtype=torch.float32)

        with torch.no_grad():
            _, _, post_curve, _ = layer(X_curve_tensor)

        y = post_curve[:, j, i].cpu().numpy()

        # Numerical derivatives with respect to the ORIGINAL feature scale.
        dy = np.gradient(y, x_raw)
        d2y = np.gradient(dy, x_raw)

        curve_df = pd.DataFrame({
            "x_raw": x_raw,
            "phi": y,
            "dphi_dx": dy,
            "d2phi_dx2": d2y,
        })

        stem = f"rank{rank+1:02d}_feature{i:02d}_{sanitize(name)}_node{j}"
        curve_df.to_csv(outdir / f"{stem}_curve.csv", index=False)

        # Function
        fig = plt.figure(figsize=(7.5, 4.8))
        ax = fig.add_subplot(111)
        ax.plot(x_raw, y)
        ax.set_xlabel(str(name))
        ax.set_ylabel(r"$\phi(x)$")
        ax.set_title(f"Outer {out_itr}, k={hp['k']}: {name} -> hidden node {j}")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(outdir / f"{stem}_phi.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

        # First derivative
        fig = plt.figure(figsize=(7.5, 4.8))
        ax = fig.add_subplot(111)
        ax.plot(x_raw, dy)
        ax.set_xlabel(str(name))
        ax.set_ylabel(r"$d\phi/dx$")
        ax.set_title(f"First derivative: {name} -> hidden node {j}")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(outdir / f"{stem}_d1.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

        # Second derivative
        fig = plt.figure(figsize=(7.5, 4.8))
        ax = fig.add_subplot(111)
        ax.plot(x_raw, d2y)
        ax.set_xlabel(str(name))
        ax.set_ylabel(r"$d^2\phi/dx^2$")
        ax.set_title(f"Second derivative: {name} -> hidden node {j}")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(outdir / f"{stem}_d2.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    print(f"\nSaved analysis to: {outdir}")


def main():
    OUTPUT_ROOT.mkdir(exist_ok=True)

    for out_itr in OUTERS:
        main_for_outer(out_itr)

    print("\nDONE")
    print(f"Root output folder: {OUTPUT_ROOT.resolve()}")


if __name__ == "__main__":
    main()
