from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


BASE = Path("KAN1L_age_local_CIF_gradients")
OUTERS = [0, 1, 2, 3, 4]
HORIZONS = [144, 288, 432]

OUT_CSV = BASE / "cross_outer_age_bin_stability.csv"
OUT_PNG = BASE / "cross_outer_age_bin_stability.png"
OUT_PNG_432 = BASE / "outer_lines_age_gradient_t432.png"


all_rows = []

for out_itr in OUTERS:
    path = BASE / f"outer_{out_itr}" / "gradient_by_real_age_bins.csv"
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path).copy()

    # The file already contains the bins ordered from youngest to oldest.
    df = df.sort_values("age_mean").reset_index(drop=True)
    df["bin_rank"] = np.arange(1, len(df) + 1)
    df["outer"] = out_itr

    all_rows.append(df)

all_df = pd.concat(all_rows, ignore_index=True)

# ------------------------------------------------------------------
# Cross-outer summary by corresponding age-bin rank.
# ------------------------------------------------------------------
summary_rows = []

for bin_rank in sorted(all_df["bin_rank"].unique()):
    block = all_df[all_df["bin_rank"] == bin_rank]

    row = {
        "bin_rank": int(bin_rank),
        "mean_age_across_outers": float(block["age_mean"].mean()),
        "sd_age_across_outers": float(block["age_mean"].std(ddof=1)),
        "n_outers": int(block["outer"].nunique()),
    }

    for h in HORIZONS:
        col = f"median_grad_t{h}"
        vals = block[col].to_numpy(dtype=float)

        row[f"mean_median_grad_t{h}"] = float(np.mean(vals))
        row[f"sd_median_grad_t{h}"] = float(np.std(vals, ddof=1))
        row[f"n_positive_outers_t{h}"] = int(np.sum(vals > 0))
        row[f"n_negative_outers_t{h}"] = int(np.sum(vals < 0))

    summary_rows.append(row)

summary = pd.DataFrame(summary_rows)
summary.to_csv(OUT_CSV, index=False)

# ------------------------------------------------------------------
# Figure 1: mean across outers ± 1 SD for all horizons.
# ------------------------------------------------------------------
fig = plt.figure(figsize=(8.5, 5.2))
ax = fig.add_subplot(111)

x = summary["mean_age_across_outers"].to_numpy()

for h in HORIZONS:
    y = summary[f"mean_median_grad_t{h}"].to_numpy()
    sd = summary[f"sd_median_grad_t{h}"].to_numpy()

    ax.plot(x, y, marker="o", label=f"t={h}")
    ax.fill_between(x, y - sd, y + sd, alpha=0.15)

ax.axhline(0.0, linewidth=1.0)
ax.set_xlabel("Mean observed age of corresponding quantile bin (years)")
ax.set_ylabel("Mean across outers of bin-median local CIF derivative")
ax.set_title("Cross-outer stability of local age sensitivity")
ax.grid(alpha=0.25)
ax.legend()
fig.tight_layout()
fig.savefig(OUT_PNG, dpi=220, bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------------------------
# Figure 2: all five outer curves at t=432.
# ------------------------------------------------------------------
fig = plt.figure(figsize=(8.5, 5.2))
ax = fig.add_subplot(111)

for out_itr in OUTERS:
    d = all_df[all_df["outer"] == out_itr].sort_values("age_mean")
    ax.plot(
        d["age_mean"],
        d["median_grad_t432"],
        marker="o",
        label=f"Outer {out_itr}"
    )

ax.axhline(0.0, linewidth=1.0)
ax.set_xlabel("Observed age-bin mean (years)")
ax.set_ylabel(r"Median local $\partial F(432|x)/\partial age$")
ax.set_title("Age sensitivity at t=432 across the five outer models")
ax.grid(alpha=0.25)
ax.legend()
fig.tight_layout()
fig.savefig(OUT_PNG_432, dpi=220, bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------------------------
# Console output
# ------------------------------------------------------------------
print("\nCROSS-OUTER AGE-BIN STABILITY")
print("=" * 100)

show_cols = [
    "bin_rank",
    "mean_age_across_outers",
]

for h in HORIZONS:
    show_cols += [
        f"mean_median_grad_t{h}",
        f"n_positive_outers_t{h}",
    ]

print(summary[show_cols].to_string(index=False))

print("\nFILES")
print(OUT_CSV)
print(OUT_PNG)
print(OUT_PNG_432)
