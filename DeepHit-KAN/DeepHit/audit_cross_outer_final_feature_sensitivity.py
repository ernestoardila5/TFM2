from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

import import_data as impt
from class_DeepHit import Model_DeepHit
from get_main_CTD_1L import get_network_settings


OUTERS = [0, 1, 2, 3, 4]
EVAL_TIMES = [144, 288, 432]
BATCH_SIZE = 64

BASE = Path("METABRIC/results_KAN1L_CTD")
FEATURE_FILE = Path("./sample data/METABRIC/cleaned_features_final.csv")
OUTDIR = Path("KAN1L_cross_outer_final_feature_sensitivity")


def load_normalizer(path):
    nf = np.load(path)
    return {
        "mode": str(nf["mode"]),
        "offset": nf["offset"],
        "scale": nf["scale"],
    }


def main():
    OUTDIR.mkdir(exist_ok=True)

    df = pd.read_csv(FEATURE_FILE)
    feature_names = list(df.columns)
    raw_data = df.to_numpy(dtype=float)

    x_dim, DATA, MASK = impt.import_dataset_METABRIC()
    mask1, _ = MASK

    if x_dim != len(feature_names):
        raise RuntimeError(
            f"x_dim={x_dim}, feature columns={len(feature_names)}"
        )

    all_idx = np.arange(len(raw_data))
    all_rows = []

    print("\n" + "=" * 120)
    print("CROSS-OUTER TOTAL ORIGINAL-FEATURE SENSITIVITY TO FINAL CIF")
    print("=" * 120)
    print(
        "Metric per feature = raw_feature_SD * "
        "RMS(d CIF_h / d raw_feature)"
    )
    print(
        "This is final-output sensitivity through the COMPLETE model, "
        "including every direct/shared/multiplicative route."
    )

    for outer in OUTERS:
        print("\n" + "#" * 120)
        print(f"OUTER {outer}")
        print("#" * 120)

        dev_idx, _ = train_test_split(
            all_idx,
            test_size=0.20,
            random_state=1234 + outer,
        )

        raw_dev = raw_data[dev_idx]

        itr_dir = BASE / f"itr_{outer}"
        hp_path = itr_dir / "hyperparameters_log.json"
        model_path = itr_dir / "models" / f"model_itr_{outer}.pt"
        norm_path = itr_dir / "models" / f"normalizer_itr_{outer}.npz"

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
            get_network_settings(hp, seed=1234 + outer),
        )
        model.load_state_dict(
            torch.load(model_path, map_location="cpu")
        )
        model.eval()

        Xn = impt.apply_normalizer(raw_data, normalizer)
        Xdev_np = Xn[dev_idx].astype(np.float32)

        raw_sd = np.std(raw_dev, axis=0, ddof=0)
        scale = np.asarray(normalizer["scale"], dtype=np.float64)

        unique_counts = np.array([
            len(np.unique(raw_dev[:, j]))
            for j in range(x_dim)
        ])

        feature_kind = np.empty(x_dim, dtype=object)
        feature_kind[unique_counts <= 2] = "binary"
        feature_kind[
            (unique_counts > 2) & (unique_counts <= 10)
        ] = "low_cardinality"
        feature_kind[unique_counts > 10] = "continuous_like"

        accum = {
            h: {
                "sum_sq": np.zeros(x_dim, dtype=np.float64),
                "sum_abs": np.zeros(x_dim, dtype=np.float64),
                "sum": np.zeros(x_dim, dtype=np.float64),
                "positive": np.zeros(x_dim, dtype=np.float64),
                "n": 0,
            }
            for h in EVAL_TIMES
        }

        n = len(Xdev_np)

        for start in range(0, n, BATCH_SIZE):
            stop = min(start + BATCH_SIZE, n)

            xb = torch.tensor(
                Xdev_np[start:stop],
                dtype=torch.float32,
                requires_grad=True,
            )

            pred = model(xb)[:, 0, :]

            for hi, h in enumerate(EVAL_TIMES):
                if h >= pred.shape[1]:
                    raise RuntimeError(
                        f"Outer {outer}: horizon {h} "
                        f">= T={pred.shape[1]}"
                    )

                Fh = pred[:, :h + 1].sum(dim=1)

                grad_u = torch.autograd.grad(
                    Fh.sum(),
                    xb,
                    retain_graph=(hi < len(EVAL_TIMES) - 1),
                    create_graph=False,
                )[0].detach().cpu().numpy()

                grad_x = grad_u / scale[None, :]

                st = accum[h]
                st["sum_sq"] += np.sum(grad_x ** 2, axis=0)
                st["sum_abs"] += np.sum(np.abs(grad_x), axis=0)
                st["sum"] += np.sum(grad_x, axis=0)
                st["positive"] += np.sum(grad_x > 0, axis=0)
                st["n"] += grad_x.shape[0]

        for h in EVAL_TIMES:
            st = accum[h]
            count = st["n"]

            rms = np.sqrt(st["sum_sq"] / count)
            mean_abs = st["sum_abs"] / count
            mean = st["sum"] / count
            positive_fraction = st["positive"] / count
            standardized = raw_sd * rms

            order = np.argsort(-standardized)
            ranks = np.empty_like(order)
            ranks[order] = np.arange(1, x_dim + 1)

            for j in range(x_dim):
                all_rows.append({
                    "outer": outer,
                    "time_index": h,
                    "feature_index": j,
                    "feature": feature_names[j],
                    "feature_kind": feature_kind[j],
                    "unique_count": int(unique_counts[j]),
                    "raw_sd": float(raw_sd[j]),
                    "RMS_dF_dx_total": float(rms[j]),
                    "mean_abs_dF_dx_total": float(mean_abs[j]),
                    "mean_dF_dx_total": float(mean[j]),
                    "positive_fraction": float(positive_fraction[j]),
                    "standardized_sensitivity_rms": float(standardized[j]),
                    "rank": int(ranks[j]),
                })

            top = pd.DataFrame([
                r for r in all_rows
                if r["outer"] == outer and r["time_index"] == h
            ]).sort_values("rank").head(10)

            print(f"\nOuter {outer} | horizon {h} | TOP 10")
            print(
                top[[
                    "rank",
                    "feature",
                    "feature_kind",
                    "standardized_sensitivity_rms",
                    "positive_fraction",
                ]].to_string(index=False)
            )

    results = pd.DataFrame(all_rows)
    results.to_csv(
        OUTDIR / "cross_outer_feature_sensitivity_all.csv",
        index=False,
    )

    # ----------------------------------------------------------
    # Cross-outer summary per horizon
    # ----------------------------------------------------------
    summary_rows = []

    for h in EVAL_TIMES:
        sub = results[results["time_index"] == h]

        for feature in feature_names:
            f = sub[sub["feature"] == feature]

            summary_rows.append({
                "time_index": h,
                "feature": feature,
                "feature_kind": f["feature_kind"].iloc[0],
                "mean_rank": float(f["rank"].mean()),
                "median_rank": float(f["rank"].median()),
                "rank_sd": float(f["rank"].std(ddof=1)),
                "n_top5": int((f["rank"] <= 5).sum()),
                "n_top10": int((f["rank"] <= 10).sum()),
                "mean_standardized_sensitivity": float(
                    f["standardized_sensitivity_rms"].mean()
                ),
                "sd_standardized_sensitivity": float(
                    f["standardized_sensitivity_rms"].std(ddof=1)
                ),
            })

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(
        OUTDIR / "cross_outer_feature_summary_by_horizon.csv",
        index=False,
    )

    # ----------------------------------------------------------
    # Aggregate all 15 outer×horizon rankings
    # ----------------------------------------------------------
    overall_rows = []

    for feature in feature_names:
        f = results[results["feature"] == feature]

        overall_rows.append({
            "feature": feature,
            "feature_kind": f["feature_kind"].iloc[0],
            "mean_rank_all_15": float(f["rank"].mean()),
            "median_rank_all_15": float(f["rank"].median()),
            "rank_sd_all_15": float(f["rank"].std(ddof=1)),
            "n_top5_all_15": int((f["rank"] <= 5).sum()),
            "n_top10_all_15": int((f["rank"] <= 10).sum()),
            "mean_standardized_sensitivity_all_15": float(
                f["standardized_sensitivity_rms"].mean()
            ),
        })

    overall = pd.DataFrame(overall_rows).sort_values(
        ["mean_rank_all_15", "rank_sd_all_15"]
    ).reset_index(drop=True)

    overall.to_csv(
        OUTDIR / "cross_outer_feature_summary_overall.csv",
        index=False,
    )

    continuous = overall[
        overall["feature_kind"] == "continuous_like"
    ].reset_index(drop=True)

    continuous.to_csv(
        OUTDIR / "cross_outer_continuous_feature_summary.csv",
        index=False,
    )

    print("\n" + "=" * 120)
    print("FINAL CROSS-OUTER SUMMARY — TOP 15 ALL FEATURES")
    print("=" * 120)
    print(overall.head(15).to_string(index=False))

    print("\nTOP CONTINUOUS-LIKE FEATURES")
    print(continuous.head(10).to_string(index=False))

    for name in ["age_at_diagnosis", "NPI", "size",
                 "lymph_nodes_positive", "lymph_nodes_removed"]:
        row = overall[overall["feature"] == name]
        if len(row):
            print("\n" + name)
            print(row.to_string(index=False))

    print("\nFILES")
    print(OUTDIR / "cross_outer_feature_sensitivity_all.csv")
    print(OUTDIR / "cross_outer_feature_summary_by_horizon.csv")
    print(OUTDIR / "cross_outer_feature_summary_overall.csv")
    print(OUTDIR / "cross_outer_continuous_feature_summary.csv")
    print("=" * 120)


if __name__ == "__main__":
    main()
