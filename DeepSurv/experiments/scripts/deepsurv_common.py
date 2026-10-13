from __future__ import print_function

import os
import re
import sys

import h5py
import numpy as np
import lasagne


# ============================================================
# LOAD ORIGINAL DEEPSURV PACKAGE
# ============================================================

sys.path.append("/DeepSurv")

from deepsurv import deep_surv


# ============================================================
# LOAD COMMON DATASET
# ============================================================

def load_dataset(filename):

    datasets = {}

    with h5py.File(filename, "r") as f:

        for split in ["train", "valid", "test"]:

            datasets[split] = {
                "x": f[split]["x"][:].astype(np.float32),
                "t": f[split]["t"][:].astype(np.float32),
                "e": f[split]["e"][:].astype(np.int32)
            }

    return datasets


# ============================================================
# INPUT
# ============================================================

if len(sys.argv) != 2:

    raise RuntimeError(
        "Usage: python deepsurv_common.py dataset.h5"
    )


filename = sys.argv[1]


# ============================================================
# DETERMINE OUTER ITERATION FROM FILENAME
# e.g. metabric_common_outer_0.h5 -> outer = 0
# ============================================================

basename = os.path.basename(filename)

match = re.search(
    r"outer_(\d+)",
    basename
)

if match is None:

    raise RuntimeError(
        "Could not determine outer iteration from filename."
    )


outer_iteration = int(
    match.group(1)
)

seed = 1234 + outer_iteration


print(
    "Outer iteration:",
    outer_iteration
)

print(
    "Seed:",
    seed
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

np.random.seed(
    seed
)

lasagne.random.set_rng(
    np.random.RandomState(seed)
)


# ============================================================
# LOAD DATA
# ============================================================

datasets = load_dataset(
    filename
)

train = datasets["train"]
valid = datasets["valid"]
test = datasets["test"]


print(
    "Train:",
    train["x"].shape
)

print(
    "Valid:",
    valid["x"].shape
)

print(
    "Test :",
    test["x"].shape
)


# ============================================================
# TRAIN-ONLY NORMALIZATION
#
# Original DeepSurv standardizes its inputs.
# We reproduce that behaviour explicitly here.
#
# Parameters are calculated ONLY using train.
#
# Constant columns are protected against division by zero.
# ============================================================

mean = train[
    "x"
].mean(
    axis=0
)

std = train[
    "x"
].std(
    axis=0
)


constant_features = np.sum(
    std == 0
)


print(
    "Constant train features:",
    int(constant_features)
)


std[
    std == 0
] = 1.0


train["x"] = (
    (
        train["x"]
        - mean
    )
    / std
).astype(
    np.float32
)


valid["x"] = (
    (
        valid["x"]
        - mean
    )
    / std
).astype(
    np.float32
)


test["x"] = (
    (
        test["x"]
        - mean
    )
    / std
).astype(
    np.float32
)


# ============================================================
# CHECK FOR INVALID VALUES
# ============================================================

for split_name, dataset in [
    ("train", train),
    ("valid", valid),
    ("test", test)
]:

    if not np.all(
        np.isfinite(
            dataset["x"]
        )
    ):

        raise RuntimeError(
            "Non-finite values found in "
            + split_name
        )


# ============================================================
# ORIGINAL DEEPSURV METABRIC HYPERPARAMETERS
#
# Original model:
# n_in = 9
#
# Our common DeepHit METABRIC:
# n_in = 79
#
# That is the only dimensional adaptation.
#
# standardize=False because we have already standardized
# explicitly using TRAIN only above.
# ============================================================

hyperparams = {

    "n_in":
        train["x"].shape[1],

    "learning_rate":
        0.0020065103592061526,

    "dropout":
        0.034404296875000004,

    "lr_decay":
        0.00055220703125,

    "momentum":
        0.8109013671875,

    "L2_reg":
        4.6593974609375,

    "batch_norm":
        False,

    "standardize":
        False,

    "hidden_layers_sizes": [
        42,
        42,
        42
    ],

    "activation":
        "selu"
}


print(
    "DeepSurv n_in:",
    hyperparams["n_in"]
)


# ============================================================
# CREATE ORIGINAL DEEPSURV MODEL
# ============================================================

model = deep_surv.DeepSurv(
    **hyperparams
)


# ============================================================
# TRAIN
# ============================================================

print(
    "\nTraining DeepSurv..."
)


metrics = model.train(

    train,

    valid_data=valid,

    n_epochs=500,

    validation_frequency=100,

    update_fn=lasagne.updates.adam,

    verbose=True
)


# ============================================================
# RESTORE MODEL WITH BEST VALIDATION LOSS
# ============================================================

best_params = metrics.get(
    "best_params",
    None
)

best_idx = metrics.get(
    "best_params_idx",
    -1
)


if best_params is not None:

    for param, value in zip(
        model.params,
        best_params
    ):

        param.set_value(
            value
        )

    print(
        "Restored best validation parameters from epoch:",
        best_idx
    )

else:

    print(
        "WARNING: no best validation parameters found."
    )


# ============================================================
# FINAL C-INDEX
# ============================================================

train_ci = model.get_concordance_index(
    **train
)

valid_ci = model.get_concordance_index(
    **valid
)

test_ci = model.get_concordance_index(
    **test
)


print(
    "\n================================"
)

print(
    "FINAL DEEPSURV RESULTS"
)

print(
    "================================"
)

print(
    "Outer:",
    outer_iteration
)

print(
    "Train C-index:",
    train_ci
)

print(
    "Valid C-index:",
    valid_ci
)

print(
    "Test C-index :",
    test_ci
)

print(
    "================================"
)