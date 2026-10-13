import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

import import_data as impt
from class_DeepHit import Model_DeepHit
from get_main_CTD_1L import get_network_settings

OUTERS = [0, 1, 2, 3, 4]
TOP_K = 10

BASE = Path("METABRIC/results_KAN1L_CTD")
FEATURE_FILE = Path("./sample data/METABRIC/cleaned_features_final.csv")
OUTDIR = Path("KAN1L_native_attribute_crosscheck")


def load_normalizer(path):
    nf = np.load(path)
    return {
        "mode": str(nf["mode"]),
        "offset": nf["offset"],
        "scale": nf["scale"],
    }


def load_outer(out_itr):
    df = pd.read_csv(FEATURE_FILE)
    feature_names = list(df.columns)
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

    return {
        "model": model,
        "Xdev": Xdev,
        "x_dim": x_dim,
        "feature_names": feature_names,
        "n_dev": len(dev_idx),
        "n_test": len(test_idx),
    }


def rank_desc(values):
    return pd.Series(values).rank(
        method="average",
        ascending=False
    ).to_numpy(dtype=float)


def analyze_outer(out_itr):
    info = load_outer(out_itr)

    model = info["model"]
    Xdev = info["Xdev"]
    x_dim = info["x_dim"]
    feature_names = info["feature_names"]

    # Real input to the cause-specific KAN.
    with torch.no_grad():
        shared_out = model.shared_kan(Xdev)
        h_cs = torch.cat([Xdev, shared_out], dim=1)

    cs_kan = model.cs_kans[0]

    # Native MultKAN activation collection.
    cs_kan.get_act(h_cs)

    # Native MultKAN attribution.
    cs_kan.attribute(plot=False)

    # Native input-node scores. First x_dim correspond to original features.
    native_input_scores = (
        cs_kan.node_scores[0]
        .detach()
        .cpu()
        .numpy()
        .reshape(-1)
    )
    native_direct = native_input_scores[:x_dim]

    # edge_actscale[0] is the standard deviation of each edge output
    # over the development patients.
    edge_scale = (
        cs_kan.edge_actscale[0]
        .detach()
        .cpu()
        .numpy()
    )

    if edge_scale.ndim != 2:
        raise RuntimeError(
            f"Unexpected edge_actscale shape: {edge_scale.shape}"
        )

    if edge_scale.shape[1] != h_cs.shape[1]:
        raise RuntimeError(
            "Unexpected edge_actscale orientation: "
            f"{edge_scale.shape}, CS input={tuple(h_cs.shape)}"
        )

    # Our previous activity metric:
    # RMS of outgoing edge standard deviations for each input.
    activity_all = np.sqrt(np.mean(edge_scale ** 2, axis=0))
    activity_direct = activity_all[:x_dim]

    rank_activity = rank_desc(activity_direct)
    rank_native = rank_desc(native_direct)

    rank_corr = float(np.corrcoef(rank_activity, rank_native)[0, 1])

    order_activity = np.argsort(-activity_direct)
    order_native = np.argsort(-native_direct)

    top_activity = set(order_activity[:TOP_K].tolist())
    top_native = set(order_native[:TOP_K].tolist())
    overlap = len(top_activity & top_native)

    rows = []
    for i, name in enumerate(feature_names):
        rows.append({
            "outer": out_itr,
            "feature_index": i,
            "feature": name,
            "activity_rms_edge_std": float(activity_direct[i]),
            "native_attribute_score": float(native_direct[i]),
            "activity_rank": float(rank_activity[i]),
            "native_rank": float(rank_native[i]),
            "top10_activity": i in top_activity,
            "top10_native": i in top_native,
            "top10_both": (i in top_activity and i in top_native),
        })

    df = pd.DataFrame(rows)

    outdir = OUTDIR / f"outer_{out_itr}"
    outdir.mkdir(parents=True, exist_ok=True)

    df.sort_values("activity_rank").to_csv(
        outdir / "feature_crosscheck.csv",
        index=False
    )

    print("\n" + "=" * 100)
    print(f"OUTER {out_itr}")
    print("=" * 100)
    print(f"Development patients: {info['n_dev']}")
    print(f"Outer test excluded: {info['n_test']}")
    print(
        f"Rank correlation (activity vs native attribute): "
        f"{rank_corr:.4f}"
    )
    print(f"Top-{TOP_K} overlap: {overlap}/{TOP_K}")

    print("\nTOP FEATURES BY OUR FUNCTIONAL ACTIVITY")
    print(
        df.sort_values("activity_rank")[
            [
                "feature",
                "activity_rms_edge_std",
                "activity_rank",
                "native_attribute_score",
                "native_rank",
            ]
        ].head(TOP_K).to_string(index=False)
    )

    print("\nTOP FEATURES BY NATIVE KAN attribute()")
    print(
        df.sort_values("native_rank")[
            [
                "feature",
                "native_attribute_score",
                "native_rank",
                "activity_rms_edge_std",
                "activity_rank",
            ]
        ].head(TOP_K).to_string(index=False)
    )

    return df, rank_corr, overlap


def main():
    OUTDIR.mkdir(exist_ok=True)

    all_frames = []
    summary_rows = []

    for out_itr in OUTERS:
        df, corr, overlap = analyze_outer(out_itr)
        all_frames.append(df)
        summary_rows.append({
            "outer": out_itr,
            "rank_correlation": corr,
            f"top{TOP_K}_overlap": overlap,
        })

    all_df = pd.concat(all_frames, ignore_index=True)
    summary_df = pd.DataFrame(summary_rows)

    all_df.to_csv(
        OUTDIR / "all_outers_feature_crosscheck.csv",
        index=False
    )
    summary_df.to_csv(
        OUTDIR / "crosscheck_summary.csv",
        index=False
    )

    recurrence = (
        all_df.groupby("feature", as_index=False)
        .agg(
            n_top10_activity=("top10_activity", "sum"),
            n_top10_native=("top10_native", "sum"),
            n_top10_both=("top10_both", "sum"),
            mean_activity_rank=("activity_rank", "mean"),
            mean_native_rank=("native_rank", "mean"),
        )
        .sort_values(
            ["n_top10_both", "n_top10_native", "n_top10_activity"],
            ascending=False,
        )
    )

    recurrence.to_csv(
        OUTDIR / "feature_recurrence_across_outers.csv",
        index=False
    )

    print("\n" + "=" * 100)
    print("CROSS-OUTER SUMMARY")
    print("=" * 100)
    print(summary_df.to_string(index=False))

    print("\nMOST RECURRENT FEATURES")
    print(recurrence.head(20).to_string(index=False))

    print("\nIMPORTANT")
    print(
        "attribute() here explains the internal cause-specific KAN representation; "
        "it does NOT include the final DeepHit Linear layer, Softmax or CIF."
    )
    print(
        "So this is a cross-check of internal KAN relevance, "
        "not a final survival-feature importance measure."
    )

    print("\nDONE")
    print("Output folder:", OUTDIR.resolve())


if __name__ == "__main__":
    main()
