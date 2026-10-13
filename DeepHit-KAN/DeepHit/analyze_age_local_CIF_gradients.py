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
OUTERS = [0, 1, 2, 3, 4]
FEATURE_NAME = "age_at_diagnosis"
HORIZONS = [144, 288, 432]

BASE = Path("METABRIC/results_KAN1L_CTD")
FEATURE_FILE = Path("./sample data/METABRIC/cleaned_features_final.csv")
OUTDIR = Path("KAN1L_age_local_CIF_gradients")

BATCH_SIZE = 256
N_AGE_BINS = 10


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

    if input_dims["num_Event"] != 1:
        raise RuntimeError(
            "This script assumes the single-event METABRIC setup."
        )

    for h in HORIZONS:
        if h >= input_dims["num_Category"]:
            raise ValueError(
                f"Horizon {h} outside num_Category={input_dims['num_Category']}"
            )

    model = Model_DeepHit(
        input_dims,
        get_network_settings(hp, seed=1234 + out_itr),
    )

    model.load_state_dict(
        torch.load(model_path, map_location="cpu")
    )
    model.eval()

    return {
        "model": model,
        "normalizer": normalizer,
        "raw_data": raw_data,
        "feature_idx": feature_idx,
        "dev_idx": dev_idx,
        "test_idx": test_idx,
    }


def local_cif_gradients(
    model,
    X_raw,
    normalizer,
    feature_idx,
):
    """
    Compute, for every REAL patient x_i and every horizon t:

        d F(t | x_i) / d age

    No covariate is replaced or perturbed.
    Gradients are evaluated exactly at each observed patient vector.

    The model receives normalized inputs, so by the chain rule:

        dF/d(age_raw) = dF/d(age_norm) * d(age_norm)/d(age_raw)
                      = dF/d(age_norm) / scale_age
    """
    X_norm = impt.apply_normalizer(
        X_raw,
        normalizer
    ).astype(np.float32)

    age_scale = float(normalizer["scale"][feature_idx])

    all_grads = {
        h: np.zeros(len(X_norm), dtype=np.float64)
        for h in HORIZONS
    }

    for start in range(0, len(X_norm), BATCH_SIZE):
        stop = min(start + BATCH_SIZE, len(X_norm))

        xb = torch.tensor(
            X_norm[start:stop],
            dtype=torch.float32,
            requires_grad=True,
        )

        pred = model(xb)                 # [B, 1, T]
        cif = torch.cumsum(
            pred[:, 0, :],
            dim=1
        )                                # [B, T]

        for j, h in enumerate(HORIZONS):
            grad = torch.autograd.grad(
                outputs=cif[:, h].sum(),
                inputs=xb,
                retain_graph=(j < len(HORIZONS) - 1),
                create_graph=False,
            )[0]

            # Because samples are independent in model.eval(), the gradient
            # in row i corresponds to dF(t|x_i)/dx_i.
            grad_age_norm = grad[:, feature_idx]

            # Convert derivative back to ORIGINAL age units.
            grad_age_raw = grad_age_norm / age_scale

            all_grads[h][start:stop] = (
                grad_age_raw.detach().cpu().numpy()
            )

    return all_grads


def summarize_gradients(df, out_itr):
    rows = []

    for h in HORIZONS:
        col = f"dCIF_dage_t{h}"
        g = df[col].to_numpy()

        rows.append({
            "outer": out_itr,
            "horizon": h,
            "n": len(g),
            "mean": float(np.mean(g)),
            "median": float(np.median(g)),
            "std": float(np.std(g, ddof=1)),
            "q05": float(np.quantile(g, 0.05)),
            "q25": float(np.quantile(g, 0.25)),
            "q75": float(np.quantile(g, 0.75)),
            "q95": float(np.quantile(g, 0.95)),
            "min": float(np.min(g)),
            "max": float(np.max(g)),
            "pct_positive": float(100.0 * np.mean(g > 0)),
            "pct_negative": float(100.0 * np.mean(g < 0)),
            "pct_near_zero": float(100.0 * np.mean(np.isclose(g, 0.0, atol=1e-8))),
        })

    return pd.DataFrame(rows)


def age_binned_summary(df, out_itr):
    tmp = df.copy()

    # Quantile bins keep a similar number of REAL patients per bin.
    tmp["age_bin"] = pd.qcut(
        tmp["age_at_diagnosis"],
        q=N_AGE_BINS,
        duplicates="drop"
    )

    rows = []

    for age_bin, group in tmp.groupby("age_bin", observed=True):
        row = {
            "outer": out_itr,
            "age_bin": str(age_bin),
            "age_mean": float(group["age_at_diagnosis"].mean()),
            "age_min": float(group["age_at_diagnosis"].min()),
            "age_max": float(group["age_at_diagnosis"].max()),
            "n": len(group),
        }

        for h in HORIZONS:
            g = group[f"dCIF_dage_t{h}"]
            row[f"median_grad_t{h}"] = float(g.median())
            row[f"mean_grad_t{h}"] = float(g.mean())
            row[f"q25_grad_t{h}"] = float(g.quantile(0.25))
            row[f"q75_grad_t{h}"] = float(g.quantile(0.75))

        rows.append(row)

    return pd.DataFrame(rows)


def analyze_outer(out_itr):
    info = load_outer(out_itr)

    model = info["model"]
    normalizer = info["normalizer"]
    raw_data = info["raw_data"]
    feature_idx = info["feature_idx"]
    dev_idx = info["dev_idx"]
    test_idx = info["test_idx"]

    # ONLY real development patients.
    X_dev_raw = raw_data[dev_idx].copy()

    grads = local_cif_gradients(
        model,
        X_dev_raw,
        normalizer,
        feature_idx,
    )

    patient_df = pd.DataFrame({
        "dataset_index": dev_idx,
        "age_at_diagnosis": X_dev_raw[:, feature_idx],
    })

    for h in HORIZONS:
        patient_df[f"dCIF_dage_t{h}"] = grads[h]

    outdir = OUTDIR / f"outer_{out_itr}"
    outdir.mkdir(parents=True, exist_ok=True)

    patient_df.to_csv(
        outdir / "patient_level_age_CIF_gradients.csv",
        index=False
    )

    summary_df = summarize_gradients(
        patient_df,
        out_itr
    )
    summary_df.to_csv(
        outdir / "gradient_summary.csv",
        index=False
    )

    binned_df = age_binned_summary(
        patient_df,
        out_itr
    )
    binned_df.to_csv(
        outdir / "gradient_by_real_age_bins.csv",
        index=False
    )

    # --------------------------------------------------------
    # Figure 1: patient-level local gradients vs REAL age
    # --------------------------------------------------------
    for h in HORIZONS:
        fig = plt.figure(figsize=(8, 5))
        ax = fig.add_subplot(111)

        ax.scatter(
            patient_df["age_at_diagnosis"],
            patient_df[f"dCIF_dage_t{h}"],
            s=10,
            alpha=0.35,
        )

        ax.axhline(0.0, linewidth=1.0)
        ax.set_xlabel("Observed age at diagnosis (years)")
        ax.set_ylabel(r"$\partial F(t|x) / \partial age$")
        ax.set_title(
            f"Outer {out_itr}: local CIF sensitivity to age at t={h}"
        )
        ax.grid(alpha=0.25)

        fig.tight_layout()
        fig.savefig(
            outdir / f"patient_gradients_vs_age_t{h}.png",
            dpi=220,
            bbox_inches="tight"
        )
        plt.close(fig)

    # --------------------------------------------------------
    # Figure 2: median local gradient by REAL-age quantile bin
    # --------------------------------------------------------
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)

    for h in HORIZONS:
        ax.plot(
            binned_df["age_mean"],
            binned_df[f"median_grad_t{h}"],
            marker="o",
            label=f"t={h}"
        )

    ax.axhline(0.0, linewidth=1.0)
    ax.set_xlabel("Mean observed age within bin (years)")
    ax.set_ylabel("Median local CIF derivative wrt age")
    ax.set_title(
        f"Outer {out_itr}: local age effect on final CIF"
    )
    ax.grid(alpha=0.25)
    ax.legend()

    fig.tight_layout()
    fig.savefig(
        outdir / "median_gradient_by_real_age_bins.png",
        dpi=220,
        bbox_inches="tight"
    )
    plt.close(fig)

    print("\n" + "=" * 92)
    print(f"OUTER {out_itr}")
    print("=" * 92)
    print(f"REAL development patients analysed: {len(dev_idx)}")
    print(f"Outer test patients excluded: {len(test_idx)}")
    print(
        f"Observed age range in development: "
        f"{patient_df['age_at_diagnosis'].min():.2f} - "
        f"{patient_df['age_at_diagnosis'].max():.2f}"
    )

    print("\nLOCAL FINAL-CIF SENSITIVITY TO AGE")
    print(
        summary_df[
            [
                "horizon",
                "mean",
                "median",
                "std",
                "q05",
                "q95",
                "pct_positive",
                "pct_negative",
            ]
        ].to_string(index=False)
    )

    print("\nMEDIAN LOCAL GRADIENT BY OBSERVED-AGE BIN")
    show_cols = ["age_mean"]
    for h in HORIZONS:
        show_cols.append(f"median_grad_t{h}")

    print(
        binned_df[show_cols].to_string(
            index=False
        )
    )

    print(f"\nSaved results in: {outdir}")

    return summary_df, binned_df


def main():
    OUTDIR.mkdir(exist_ok=True)

    all_summary = []
    all_binned = []

    for out_itr in OUTERS:
        summary_df, binned_df = analyze_outer(out_itr)
        all_summary.append(summary_df)
        all_binned.append(binned_df)

    summary_all = pd.concat(
        all_summary,
        ignore_index=True
    )
    binned_all = pd.concat(
        all_binned,
        ignore_index=True
    )

    summary_all.to_csv(
        OUTDIR / "all_outers_gradient_summary.csv",
        index=False
    )
    binned_all.to_csv(
        OUTDIR / "all_outers_gradient_by_real_age_bins.csv",
        index=False
    )

    # --------------------------------------------------------
    # Cross-outer summary
    # --------------------------------------------------------
    cross_rows = []

    for h in HORIZONS:
        s = summary_all[
            summary_all["horizon"] == h
        ]

        cross_rows.append({
            "horizon": h,
            "mean_of_outer_mean_gradients": float(s["mean"].mean()),
            "sd_of_outer_mean_gradients": float(s["mean"].std(ddof=1)),
            "mean_of_outer_median_gradients": float(s["median"].mean()),
            "mean_pct_positive": float(s["pct_positive"].mean()),
            "mean_pct_negative": float(s["pct_negative"].mean()),
        })

    cross_df = pd.DataFrame(cross_rows)
    cross_df.to_csv(
        OUTDIR / "cross_outer_summary.csv",
        index=False
    )

    print("\n" + "=" * 92)
    print("CROSS-OUTER SUMMARY")
    print("=" * 92)
    print(cross_df.to_string(index=False))
    print("\nDONE")
    print("Output folder:", OUTDIR.resolve())
    print("=" * 92)


if __name__ == "__main__":
    main()
