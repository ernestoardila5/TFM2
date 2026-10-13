import sys
import h5py
import numpy as np
import pandas as pd

from scipy.linalg import qr

from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index


EVAL_TIMES = [144, 288, 432]


def load_data(filename):
    data = {}

    with h5py.File(filename, "r") as f:
        for split in ["train", "valid", "test"]:
            data[split] = {
                "x": f[split]["x"][:],
                "t": f[split]["t"][:],
                "e": f[split]["e"][:]
            }

    return data


def censoring_prob(y, t):

    t = t.reshape(-1)
    y = y.reshape(-1)

    kmf = KaplanMeierFitter()

    kmf.fit(
        t,
        event_observed=(y == 0).astype(int)
    )

    G = np.asarray(
        kmf.survival_function_.reset_index()
    ).transpose()

    nonzero = G[1, :] != 0

    if np.any(nonzero):
        G[1, G[1, :] == 0] = G[1, nonzero][-1]

    return G


def weighted_c_index(
    t_train,
    y_train,
    prediction,
    t_test,
    y_test,
    time
):

    G = censoring_prob(
        y_train,
        t_train
    )

    n = len(prediction)

    A = np.zeros((n, n))
    Q = np.zeros((n, n))
    N_t = np.zeros((n, n))

    for i in range(n):

        tmp_idx = np.where(
            G[0, :] >= t_test[i]
        )[0]

        if len(tmp_idx) == 0:
            W = (1.0 / G[1, -1]) ** 2
        else:
            W = (
                1.0 / G[1, tmp_idx[0]]
            ) ** 2

        A[
            i,
            np.where(t_test[i] < t_test)
        ] = W

        Q[
            i,
            np.where(prediction[i] > prediction)
        ] = 1.0

        if (
            t_test[i] <= time
            and y_test[i] == 1
        ):
            N_t[i, :] = 1.0

    num = np.sum(
        (A * N_t) * Q
    )

    den = np.sum(
        A * N_t
    )

    if den == 0:
        return -1

    return float(
        num / den
    )


if len(sys.argv) != 2:
    raise RuntimeError(
        "Usage: python cph_common.py dataset.h5"
    )


filename = sys.argv[1]

data = load_data(filename)


x_train = data["train"]["x"].astype(np.float64)
t_train = data["train"]["t"].astype(np.float64)
e_train = data["train"]["e"].astype(np.int32)

x_valid = data["valid"]["x"].astype(np.float64)
t_valid = data["valid"]["t"].astype(np.float64)
e_valid = data["valid"]["e"].astype(np.int32)

x_test = data["test"]["x"].astype(np.float64)
t_test = data["test"]["t"].astype(np.float64)
e_test = data["test"]["e"].astype(np.int32)


print("Original features:", x_train.shape[1])
print("Train:", len(x_train))
print("Valid:", len(x_valid))
print("Test :", len(x_test))


# ============================================================
# 1. Remove constant features using TRAIN only
# ============================================================

std = np.std(
    x_train,
    axis=0
)

nonconstant = np.where(
    std > 1e-12
)[0]

x_train = x_train[:, nonconstant]
x_valid = x_valid[:, nonconstant]
x_test = x_test[:, nonconstant]

print(
    "After removing constants:",
    x_train.shape[1]
)


# ============================================================
# 2. Normalize using TRAIN only
# ============================================================

mean = np.mean(
    x_train,
    axis=0
)

std = np.std(
    x_train,
    axis=0
)

std[
    std < 1e-12
] = 1.0

x_train = (
    x_train - mean
) / std

x_valid = (
    x_valid - mean
) / std

x_test = (
    x_test - mean
) / std


# ============================================================
# 3. Remove exact linear dependencies
#
# QR pivoting chooses a full-rank subset.
# This removes redundant one-hot columns but does not remove
# independent information.
# ============================================================

Qmat, R, piv = qr(
    x_train,
    mode="economic",
    pivoting=True
)

diag = np.abs(
    np.diag(R)
)

if len(diag) == 0:
    raise RuntimeError(
        "Empty design matrix."
    )

tol = (
    max(x_train.shape)
    * np.finfo(float).eps
    * diag[0]
)

rank = np.sum(
    diag > tol
)

keep = np.sort(
    piv[:rank]
)

x_train = x_train[:, keep]
x_valid = x_valid[:, keep]
x_test = x_test[:, keep]


print(
    "Independent features:",
    x_train.shape[1]
)

print(
    "Redundant features removed:",
    len(nonconstant) - x_train.shape[1]
)


# ============================================================
# 4. Fit Cox PH
# ============================================================

train_df = pd.DataFrame(
    x_train
)

train_df["time"] = t_train
train_df["event"] = e_train


print("\nTraining Cox PH...")


cox = CoxPHFitter()

cox.fit(
    train_df,
    duration_col="time",
    event_col="event",
    include_likelihood=True
)


# ============================================================
# 5. Predictions
# ============================================================

train_risk = np.squeeze(
    cox.predict_partial_hazard(
        pd.DataFrame(x_train)
    ).values
)

valid_risk = np.squeeze(
    cox.predict_partial_hazard(
        pd.DataFrame(x_valid)
    ).values
)

test_risk = np.squeeze(
    cox.predict_partial_hazard(
        pd.DataFrame(x_test)
    ).values
)


# ============================================================
# 6. Standard Harrell C-index - sanity check
# ============================================================

train_ci = concordance_index(
    t_train,
    -train_risk,
    e_train
)

valid_ci = concordance_index(
    t_valid,
    -valid_risk,
    e_valid
)

test_ci = concordance_index(
    t_test,
    -test_risk,
    e_test
)


print("\nHarrell C-index")
print("Train:", train_ci)
print("Valid:", valid_ci)
print("Test :", test_ci)


# ============================================================
# 7. SAME weighted C(t)-index used by DeepHit pipeline
# ============================================================

print("\nWeighted C-index on OUTER TEST")

for horizon in EVAL_TIMES:

    ci = weighted_c_index(
        t_train,
        e_train,
        test_risk,
        t_test,
        e_test,
        horizon
    )

    print(
        "Time %d: %.6f"
        % (
            horizon,
            ci
        )
    )