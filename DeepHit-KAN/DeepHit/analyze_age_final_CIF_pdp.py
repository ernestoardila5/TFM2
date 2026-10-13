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
OUTERS = [2, 4]
FEATURE_NAME = "age_at_diagnosis"

BASE = Path("METABRIC/results_KAN1L_CTD")
FEATURE_FILE = Path("./sample data/METABRIC/cleaned_features_final.csv")
OUTDIR = Path("KAN1L_age_final_prediction_pdp")

# Same METABRIC horizons previously used as secondary time points.
HORIZONS = [144, 288, 432]

# Ages for full mean-CIF curves.
CURVE_AGES = [40.0, 55.0, 70.0, 85.0]

# Number of ages for the partial-dependence curve.
N_AGE_GRID = 60

# Forward batch size.
BATCH_SIZE = 512


def load_normalizer(path):
    nf = np.load(path)
    return {
        "mode": str(nf["mode"]),
        "offset": nf["offset"],
        "scale": nf["scale"],
    }


def predict_in_batches(model, x_np, batch_size=BATCH_SIZE):
    preds = []

    with torch.no_grad():
        for start in range(0, len(x_np), batch_size):
            stop = min(start + batch_size, len(x_np))
            xb = torch.tensor(
                x_np[start:stop],
                dtype=torch.float32
            )
            yb = model(xb).cpu().numpy()
            preds.append(yb)

    return np.concatenate(preds, axis=0)


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

    model = Model_DeepHit(
        input_dims,
        get_network_settings(hp, seed=1234 + out_itr),
    )
    model.load_state_dict(
        torch.load(model_path, map_location="cpu")
    )
    model.eval()

    if input_dims["num_Event"] != 1:
        raise RuntimeError(
            "This script is written for the single-event METABRIC setup."
        )

    for h in HORIZONS:
        if h >= input_dims["num_Category"]:
            raise ValueError(
                f"Horizon {h} outside num_Category={input_dims['num_Category']}"
            )

    return {
        "model": model,
        "normalizer": normalizer,
        "raw_data": raw_data,
        "feature_idx": feature_idx,
        "dev_idx": dev_idx,
        "test_idx": test_idx,
        "num_category": input_dims["num_Category"],
    }


def mean_cif_for_forced_age(
    model,
    normalizer,
    raw_dev,
    feature_idx,
    age_value,
):
    """
    Partial-dependence style intervention:
    set age_at_diagnosis = age_value for EVERY development patient,
    leave all other covariates unchanged, run the FULL DeepHit-KAN,
    and average the resulting CIF across patients.
    """
    x_mod = raw_dev.copy()
    x_mod[:, feature_idx] = age_value

    x_norm = impt.apply_normalizer(
        x_mod,
        normalizer
    )

    pred = predict_in_batches(
        model,
        x_norm
    )

    # Single event:
    # CIF(t|x) = cumulative sum_tau P(T=tau | x)
    cif = np.cumsum(
        pred[:, 0, :],
        axis=1
    )

    return np.mean(cif, axis=0)


def analyze_outer(out_itr):
    info = load_outer(out_itr)

    model = info["model"]
    normalizer = info["normalizer"]
    raw_data = info["raw_data"]
    feature_idx = info["feature_idx"]
    dev_idx = info["dev_idx"]
    test_idx = info["test_idx"]
    num_category = info["num_category"]

    raw_dev = raw_data[dev_idx].copy()

    age_dev = raw_dev[:, feature_idx]
    age_dev = age_dev[np.isfinite(age_dev)]

    lo = float(np.quantile(age_dev, 0.01))
    hi = float(np.quantile(age_dev, 0.99))
    age_grid = np.linspace(lo, hi, N_AGE_GRID)

    outdir = OUTDIR / f"outer_{out_itr}"
    outdir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # 1) Mean final CIF at fixed horizons as a function of age
    # --------------------------------------------------------
    pd_rows = []

    print("\n" + "=" * 90)
    print(f"OUTER {out_itr}")
    print("=" * 90)
    print(f"Development patients used: {len(dev_idx)}")
    print(f"Outer test patients excluded: {len(test_idx)}")
    print(f"Age range used (DEV 1%-99%): {lo:.3f} to {hi:.3f}")

    for n, age_value in enumerate(age_grid, start=1):
        mean_cif = mean_cif_for_forced_age(
            model,
            normalizer,
            raw_dev,
            feature_idx,
            age_value,
        )

        row = {"age": float(age_value)}
        for h in HORIZONS:
            row[f"mean_CIF_{h}"] = float(mean_cif[h])
        pd_rows.append(row)

        if n == 1 or n == len(age_grid) or n % 10 == 0:
            print(
                f"  age grid {n:>2}/{len(age_grid)} "
                f"(age={age_value:.2f})"
            )

    pd_df = pd.DataFrame(pd_rows)
    pd_df.to_csv(
        outdir / "age_partial_dependence_horizons.csv",
        index=False
    )

    # Plot all selected horizons in one figure.
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)

    for h in HORIZONS:
        ax.plot(
            pd_df["age"],
            pd_df[f"mean_CIF_{h}"],
            label=f"t={h}"
        )

    ax.set_xlabel("Age at diagnosis (years)")
    ax.set_ylabel("Mean predicted CIF")
    ax.set_title(
        f"Outer {out_itr}: final DeepHit-KAN prediction vs age"
    )
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        outdir / "age_partial_dependence_horizons.png",
        dpi=220,
        bbox_inches="tight"
    )
    plt.close(fig)

    # --------------------------------------------------------
    # 2) Full mean CIF curves for a few concrete ages
    # --------------------------------------------------------
    curve_rows = []
    curve_dict = {}

    for age_value in CURVE_AGES:
        if not (lo <= age_value <= hi):
            continue

        mean_cif = mean_cif_for_forced_age(
            model,
            normalizer,
            raw_dev,
            feature_idx,
            age_value,
        )

        curve_dict[age_value] = mean_cif

        for t, value in enumerate(mean_cif):
            curve_rows.append({
                "age": age_value,
                "time": t,
                "mean_CIF": float(value),
            })

    curves_df = pd.DataFrame(curve_rows)
    curves_df.to_csv(
        outdir / "mean_CIF_curves_selected_ages.csv",
        index=False
    )

    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)

    for age_value, mean_cif in curve_dict.items():
        ax.plot(
            np.arange(num_category),
            mean_cif,
            label=f"age={age_value:.0f}"
        )

    ax.set_xlabel("Discrete survival time")
    ax.set_ylabel("Mean predicted CIF")
    ax.set_title(
        f"Outer {out_itr}: mean CIF curves at selected ages"
    )
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        outdir / "mean_CIF_curves_selected_ages.png",
        dpi=220,
        bbox_inches="tight"
    )
    plt.close(fig)

    # --------------------------------------------------------
    # 3) Simple descriptive summary of final prediction effect
    # --------------------------------------------------------
    print("\nFINAL-PREDICTION AGE EFFECT AT SELECTED HORIZONS")

    for h in HORIZONS:
        y = pd_df[f"mean_CIF_{h}"].to_numpy()
        diffs = np.diff(y)

        n_pos = int(np.sum(diffs > 0))
        n_neg = int(np.sum(diffs < 0))
        n_zero = int(np.sum(np.isclose(diffs, 0)))

        min_idx = int(np.argmin(y))
        max_idx = int(np.argmax(y))

        print(f"\nHorizon t={h}")
        print(
            f"  CIF range over age grid: "
            f"{y[min_idx]:.6f} to {y[max_idx]:.6f}"
        )
        print(
            f"  Minimum at age ~{pd_df['age'].iloc[min_idx]:.2f}"
        )
        print(
            f"  Maximum at age ~{pd_df['age'].iloc[max_idx]:.2f}"
        )
        print(
            f"  Adjacent age steps: "
            f"{n_pos} increasing, {n_neg} decreasing, {n_zero} ~flat"
        )

        # Compare 40 vs 80 if both are inside range.
        if lo <= 40 <= hi and lo <= 80 <= hi:
            cif_40 = np.interp(
                40.0,
                pd_df["age"],
                y
            )
            cif_80 = np.interp(
                80.0,
                pd_df["age"],
                y
            )
            print(
                f"  Approx. mean CIF(age=40): {cif_40:.6f}"
            )
            print(
                f"  Approx. mean CIF(age=80): {cif_80:.6f}"
            )
            print(
                f"  Difference 80 - 40: {cif_80 - cif_40:+.6f}"
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
