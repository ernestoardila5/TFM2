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
EVAL_TIMES = [144, 288, 432]
BATCH_SIZE = 64

BASE = Path("METABRIC/results_KAN1L_CTD")
FEATURE_FILE = Path("./sample data/METABRIC/cleaned_features_final.csv")
OUTDIR = Path("KAN1L_total_input_CIF_sensitivity_outer2")


def load_normalizer(path):
    nf = np.load(path)
    return {
        "mode": str(nf["mode"]),
        "offset": nf["offset"],
        "scale": nf["scale"],
    }


def main():
    OUTDIR.mkdir(exist_ok=True)

    # --------------------------------------------------------
    # Data and exact OUTER-2 development split
    # --------------------------------------------------------
    df = pd.read_csv(FEATURE_FILE)
    feature_names = list(df.columns)
    raw_data = df.to_numpy(dtype=float)

    x_dim, DATA, MASK = impt.import_dataset_METABRIC()
    mask1, _ = MASK

    if len(feature_names) != x_dim:
        raise RuntimeError(
            f"Feature-name count ({len(feature_names)}) != x_dim ({x_dim})"
        )

    all_idx = np.arange(len(raw_data))
    dev_idx, _ = train_test_split(
        all_idx,
        test_size=0.20,
        random_state=1234 + OUTER,
    )

    raw_dev = raw_data[dev_idx]

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

    if input_dims["num_Event"] != 1:
        raise RuntimeError("This script assumes single-event METABRIC.")

    model = Model_DeepHit(
        input_dims,
        get_network_settings(hp, seed=1234 + OUTER),
    )
    model.load_state_dict(
        torch.load(model_path, map_location="cpu")
    )
    model.eval()

    Xn = impt.apply_normalizer(raw_data, normalizer)
    Xdev_np = Xn[dev_idx].astype(np.float32)

    n = len(Xdev_np)
    d = Xdev_np.shape[1]

    # --------------------------------------------------------
    # Feature metadata
    # --------------------------------------------------------
    raw_sd = np.std(raw_dev, axis=0, ddof=0)
    unique_counts = np.array([
        len(np.unique(raw_dev[:, j][~np.isnan(raw_dev[:, j])]))
        for j in range(d)
    ])

    feature_kind = np.empty(d, dtype=object)
    feature_kind[unique_counts <= 2] = "binary"
    feature_kind[(unique_counts > 2) & (unique_counts <= 10)] = "low_cardinality"
    feature_kind[unique_counts > 10] = "continuous_like"

    scale = np.asarray(normalizer["scale"], dtype=float)
    if np.any(np.abs(scale) < 1e-12):
        bad = np.where(np.abs(scale) < 1e-12)[0]
        raise RuntimeError(
            f"Normalizer has near-zero scale for feature indices {bad.tolist()}"
        )

    # --------------------------------------------------------
    # Accumulators for TOTAL derivative dF_h / dx_i
    #
    # Important: this derivative traverses ALL routes:
    #   x_i -> CS directly
    #   x_i -> Shared -> CS
    #   x_i -> any other node -> final softmax/CIF
    # --------------------------------------------------------
    stats = {}
    for h in EVAL_TIMES:
        stats[h] = {
            "sum_sq": np.zeros(d, dtype=np.float64),
            "sum_abs": np.zeros(d, dtype=np.float64),
            "sum": np.zeros(d, dtype=np.float64),
            "positive": np.zeros(d, dtype=np.float64),
            "n": 0,
        }

    for start in range(0, n, BATCH_SIZE):
        stop = min(start + BATCH_SIZE, n)

        xb = torch.tensor(
            Xdev_np[start:stop],
            dtype=torch.float32,
            requires_grad=True,
        )

        pred = model(xb)[:, 0, :]  # [B, T]

        for h_idx, h in enumerate(EVAL_TIMES):
            if h >= pred.shape[1]:
                raise RuntimeError(
                    f"Horizon {h} outside model output dimension {pred.shape[1]}"
                )

            Fh = pred[:, :h + 1].sum(dim=1)

            grad_u = torch.autograd.grad(
                Fh.sum(),
                xb,
                retain_graph=(h_idx < len(EVAL_TIMES) - 1),
                create_graph=False,
            )[0].detach().cpu().numpy()

            # u = (x - offset) / scale
            # therefore dF/dx = dF/du * du/dx = dF/du / scale
            grad_x = grad_u / scale[None, :]

            st = stats[h]
            st["sum_sq"] += np.sum(grad_x ** 2, axis=0)
            st["sum_abs"] += np.sum(np.abs(grad_x), axis=0)
            st["sum"] += np.sum(grad_x, axis=0)
            st["positive"] += np.sum(grad_x > 0, axis=0)
            st["n"] += grad_x.shape[0]

        print(f"Processed {stop}/{n} patients")

    # --------------------------------------------------------
    # Build ranked tables
    # --------------------------------------------------------
    all_tables = []

    print("\n" + "=" * 110)
    print("TOTAL ORIGINAL-FEATURE SENSITIVITY TO FINAL CIF — OUTER 2")
    print("=" * 110)
    print(f"Development patients: {n}")
    print(f"Original model inputs: {d}")
    print(
        "IMPORTANT: these are TOTAL derivatives through the whole network, "
        "not one selected path."
    )
    print(
        "For binary/low-cardinality inputs, the derivative is mathematically "
        "valid for the continuous encoding used by the network, but should not "
        "be interpreted as an infinitesimal real-world change."
    )

    for h in EVAL_TIMES:
        st = stats[h]
        count = st["n"]

        rms = np.sqrt(st["sum_sq"] / count)
        mean_abs = st["sum_abs"] / count
        mean = st["sum"] / count
        positive_fraction = st["positive"] / count

        standardized = raw_sd * rms

        table = pd.DataFrame({
            "time_index": h,
            "feature_index": np.arange(d),
            "feature": feature_names,
            "feature_kind": feature_kind,
            "unique_count": unique_counts,
            "raw_sd": raw_sd,
            "RMS_dF_dx_total": rms,
            "mean_abs_dF_dx_total": mean_abs,
            "mean_dF_dx_total": mean,
            "positive_fraction": positive_fraction,
            "standardized_sensitivity_rms": standardized,
        })

        table = table.sort_values(
            "standardized_sensitivity_rms",
            ascending=False,
        ).reset_index(drop=True)

        table.insert(
            0,
            "rank_all",
            np.arange(1, len(table) + 1),
        )

        # Rank continuous-like variables separately
        cont = table[
            table["feature_kind"] == "continuous_like"
        ].copy()
        cont = cont.sort_values(
            "standardized_sensitivity_rms",
            ascending=False,
        ).reset_index(drop=True)
        cont.insert(
            0,
            "rank_continuous_like",
            np.arange(1, len(cont) + 1),
        )

        print("\n" + "-" * 110)
        print(f"CIF HORIZON t = {h}")
        print("-" * 110)

        print("\nTOP 15 FEATURES — TOTAL STANDARDIZED SENSITIVITY")
        print(
            table.loc[:, [
                "rank_all",
                "feature_index",
                "feature",
                "feature_kind",
                "unique_count",
                "standardized_sensitivity_rms",
                "RMS_dF_dx_total",
                "mean_dF_dx_total",
                "positive_fraction",
            ]].head(15).to_string(index=False)
        )

        print("\nTOP 10 CONTINUOUS-LIKE FEATURES")
        print(
            cont.loc[:, [
                "rank_continuous_like",
                "feature_index",
                "feature",
                "unique_count",
                "standardized_sensitivity_rms",
                "RMS_dF_dx_total",
                "mean_dF_dx_total",
                "positive_fraction",
            ]].head(10).to_string(index=False)
        )

        age_row = table[table["feature"] == "age_at_diagnosis"]
        if len(age_row):
            r = age_row.iloc[0]
            print("\nAGE_AT_DIAGNOSIS")
            print(
                f"rank_all={int(r['rank_all'])}/{d} | "
                f"kind={r['feature_kind']} | "
                f"unique={int(r['unique_count'])} | "
                f"std_sensitivity={r['standardized_sensitivity_rms']:.10f} | "
                f"RMS(dF/dx)={r['RMS_dF_dx_total']:.10f} | "
                f"mean(dF/dx)={r['mean_dF_dx_total']:.10f} | "
                f"positive_fraction={r['positive_fraction']:.4f}"
            )

        table.to_csv(
            OUTDIR / f"features_ranked_total_CIF_sensitivity_t{h}.csv",
            index=False,
        )
        cont.to_csv(
            OUTDIR / f"continuous_like_ranked_t{h}.csv",
            index=False,
        )

        all_tables.append(table)

    combined = pd.concat(all_tables, ignore_index=True)
    combined.to_csv(
        OUTDIR / "all_horizons_total_feature_sensitivity.csv",
        index=False,
    )

    print("\nFILES")
    for h in EVAL_TIMES:
        print(
            OUTDIR / f"features_ranked_total_CIF_sensitivity_t{h}.csv"
        )
        print(
            OUTDIR / f"continuous_like_ranked_t{h}.csv"
        )
    print(
        OUTDIR / "all_horizons_total_feature_sensitivity.csv"
    )
    print("=" * 110)


if __name__ == "__main__":
    main()
