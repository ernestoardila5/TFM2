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


OUTERS = [2, 4]
TOP_FEATURES = 10
TOP_PLOTS = 5

BASE = Path("METABRIC/results_KAN1L_CTD")
FEATURE_FILE = Path("./sample data/METABRIC/cleaned_features_final.csv")
OUTPUT_ROOT = Path("KAN1L_function_analysis_v2")


def sanitize(name):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name))[:120]


def load_normalizer(path):
    nf = np.load(path)
    return {
        "mode": str(nf["mode"]),
        "offset": nf["offset"],
        "scale": nf["scale"],
    }


def is_binary_or_discrete(values):
    values = values[np.isfinite(values)]
    unique = np.unique(values)
    return len(unique) <= 5, unique


def edge_activity(postacts, n_inputs):
    # postacts: [N, out_dim, in_dim]
    edge_std = torch.std(postacts, dim=0, unbiased=False).cpu().numpy()
    edge_std = edge_std[:, :n_inputs]
    feat_activity = np.sqrt(np.mean(edge_std ** 2, axis=0))
    return edge_std, feat_activity


def rank_inputs(edge_std, feat_activity, feature_names):
    rows = []
    for i, name in enumerate(feature_names):
        j = int(np.argmax(edge_std[:, i]))
        rows.append({
            "feature_index": i,
            "feature_name": name,
            "activity_rms_edge_std": float(feat_activity[i]),
            "strongest_hidden_node": j,
            "strongest_edge_std": float(edge_std[j, i]),
        })
    return pd.DataFrame(rows).sort_values(
        "activity_rms_edge_std", ascending=False
    ).reset_index(drop=True)


def plot_edge_function(
    layer,
    input_index,
    output_index,
    raw_values,
    normalizer,
    feature_index,
    feature_name,
    outdir,
    prefix,
    k
):
    vals = raw_values[np.isfinite(raw_values)]
    discrete, unique = is_binary_or_discrete(vals)

    if discrete:
        # Evaluate only observed discrete values. Derivatives are not meaningful here.
        x_raw = unique.astype(float)
        x_norm = (
            (x_raw - normalizer["offset"][feature_index])
            / normalizer["scale"][feature_index]
        )

        in_dim = layer.in_dim
        H = torch.zeros((len(x_norm), in_dim), dtype=torch.float32)
        H[:, input_index] = torch.tensor(x_norm, dtype=torch.float32)

        with torch.no_grad():
            _, _, post, _ = layer(H)
        y = post[:, output_index, input_index].cpu().numpy()

        df = pd.DataFrame({"x_raw": x_raw, "phi": y})
        stem = f"{prefix}_{sanitize(feature_name)}_node{output_index}"
        df.to_csv(outdir / f"{stem}_discrete.csv", index=False)

        fig = plt.figure(figsize=(7.2, 4.6))
        ax = fig.add_subplot(111)
        ax.plot(x_raw, y, marker="o")
        ax.set_xlabel(str(feature_name))
        ax.set_ylabel(r"$\phi(x)$")
        ax.set_title(f"{feature_name} -> hidden node {output_index} (k={k}, discrete)")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(outdir / f"{stem}_phi.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

        if len(x_raw) == 2:
            delta = float(y[1] - y[0])
            print(f"    discrete delta phi({x_raw[1]})-phi({x_raw[0]}) = {delta:.6f}")
        return

    lo = float(np.quantile(vals, 0.01))
    hi = float(np.quantile(vals, 0.99))
    if hi <= lo:
        lo = float(np.min(vals))
        hi = float(np.max(vals))
    if hi <= lo:
        print(f"    skipping {feature_name}: constant")
        return

    x_raw = np.linspace(lo, hi, 500)
    x_norm = (
        (x_raw - normalizer["offset"][feature_index])
        / normalizer["scale"][feature_index]
    )

    in_dim = layer.in_dim
    H = torch.zeros((len(x_norm), in_dim), dtype=torch.float32)
    H[:, input_index] = torch.tensor(x_norm, dtype=torch.float32)

    with torch.no_grad():
        _, _, post, _ = layer(H)
    y = post[:, output_index, input_index].cpu().numpy()

    dy = np.gradient(y, x_raw)
    d2y = np.gradient(dy, x_raw)

    df = pd.DataFrame({
        "x_raw": x_raw,
        "phi": y,
        "dphi_dx": dy,
        "d2phi_dx2": d2y,
    })

    stem = f"{prefix}_{sanitize(feature_name)}_node{output_index}"
    df.to_csv(outdir / f"{stem}_curve.csv", index=False)

    fig = plt.figure(figsize=(7.2, 4.6))
    ax = fig.add_subplot(111)
    ax.plot(x_raw, y)
    ax.set_xlabel(str(feature_name))
    ax.set_ylabel(r"$\phi(x)$")
    ax.set_title(f"{feature_name} -> hidden node {output_index} (k={k})")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / f"{stem}_phi.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure(figsize=(7.2, 4.6))
    ax = fig.add_subplot(111)
    ax.plot(x_raw, dy)
    ax.set_xlabel(str(feature_name))
    ax.set_ylabel(r"$d\phi/dx$")
    ax.set_title(f"First derivative: {feature_name} (k={k})")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / f"{stem}_d1.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure(figsize=(7.2, 4.6))
    ax = fig.add_subplot(111)
    ax.plot(x_raw, d2y)
    ax.set_xlabel(str(feature_name))
    ax.set_ylabel(r"$d^2\phi/dx^2$")
    ax.set_title(f"Second derivative: {feature_name} (k={k})")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / f"{stem}_d2.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def analyze_outer(out_itr):
    print("\n" + "=" * 78)
    print(f"OUTER {out_itr}")
    print("=" * 78)

    df = pd.read_csv(FEATURE_FILE)
    feature_names = list(df.columns)
    raw_data = df.to_numpy(dtype=float)

    x_dim, DATA, MASK = impt.import_dataset_METABRIC()
    _, _, _ = DATA
    mask1, _ = MASK

    itr_dir = BASE / f"itr_{out_itr}"
    hp_path = itr_dir / "hyperparameters_log.json"
    model_path = itr_dir / "models" / f"model_itr_{out_itr}.pt"
    norm_path = itr_dir / "models" / f"normalizer_itr_{out_itr}.npz"

    with hp_path.open("r", encoding="utf-8") as f:
        hp = json.load(f)

    normalizer = load_normalizer(norm_path)
    Xn = impt.apply_normalizer(raw_data, normalizer)

    input_dims = {
        "x_dim": x_dim,
        "num_Event": mask1.shape[1],
        "num_Category": mask1.shape[2],
    }

    model = Model_DeepHit(
        input_dims,
        get_network_settings(hp, seed=1234 + out_itr)
    )
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    X = torch.tensor(Xn, dtype=torch.float32)

    with torch.no_grad():
        shared_out = model.shared_kan(X)
        h = torch.cat([X, shared_out], dim=1)

    outdir = OUTPUT_ROOT / f"outer_{out_itr}_k{hp['k']}"
    outdir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------
    # 1) SHARED KAN diagnostics
    # ------------------------------------------------------------
    shared_layer = model.shared_kan.act_fun[0]
    with torch.no_grad():
        _, _, shared_post, _ = shared_layer(X)

    s_edge_std, s_activity = edge_activity(shared_post, x_dim)
    shared_rank = rank_inputs(s_edge_std, s_activity, feature_names)
    shared_rank.to_csv(outdir / "shared_feature_activity_ranking.csv", index=False)

    shared_absmax = float(torch.max(torch.abs(shared_out)).cpu())
    shared_std = float(torch.std(shared_out, unbiased=False).cpu())

    print(f"k = {hp['k']}")
    print(f"SHARED output |max| = {shared_absmax:.6e}")
    print(f"SHARED output std   = {shared_std:.6e}")

    print("\nTOP SHARED FEATURES")
    print(shared_rank.head(TOP_FEATURES).to_string(index=False))

    # ------------------------------------------------------------
    # 2) CAUSE-SPECIFIC KAN: direct edges from ORIGINAL x variables
    # Since METABRIC has one event, there is one CS KAN.
    # h = [x, shared_out], so the first x_dim CS inputs are original x.
    # ------------------------------------------------------------
    cs_layer = model.cs_kans[0].act_fun[0]
    with torch.no_grad():
        _, _, cs_post, _ = cs_layer(h)

    cs_edge_std, cs_activity = edge_activity(cs_post, x_dim)
    cs_rank = rank_inputs(cs_edge_std, cs_activity, feature_names)
    cs_rank.to_csv(outdir / "cs_direct_feature_activity_ranking.csv", index=False)

    print("\nTOP CAUSE-SPECIFIC DIRECT ORIGINAL FEATURES")
    print(cs_rank.head(TOP_FEATURES).to_string(index=False))

    # ------------------------------------------------------------
    # 3) Curves for top CS-direct original features
    # ------------------------------------------------------------
    print("\nCURVES FOR TOP CS-DIRECT FEATURES")
    for rank, row in cs_rank.head(TOP_PLOTS).iterrows():
        i = int(row["feature_index"])
        j = int(row["strongest_hidden_node"])
        name = row["feature_name"]
        print(f"  {rank+1}. {name} -> node {j}")

        plot_edge_function(
            layer=cs_layer,
            input_index=i,         # first x_dim entries of h are original x
            output_index=j,
            raw_values=raw_data[:, i],
            normalizer=normalizer,
            feature_index=i,
            feature_name=name,
            outdir=outdir,
            prefix=f"rank{rank+1:02d}_csdirect",
            k=hp["k"],
        )

    print(f"\nSaved: {outdir}")


def main():
    OUTPUT_ROOT.mkdir(exist_ok=True)
    for out_itr in OUTERS:
        analyze_outer(out_itr)
    print("\nDONE")
    print(f"Root output folder: {OUTPUT_ROOT.resolve()}")


if __name__ == "__main__":
    main()
