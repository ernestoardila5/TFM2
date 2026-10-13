import os
import json
import random

import numpy as np
import torch

import import_data_MLP_FINAL as impt

from class_DeepHit_MLP_FINAL import Model_DeepHit
from get_main_MLP_FINAL import get_valid_performance, get_network_settings
from utils_eval_MLP_FINAL import weighted_c_index

from sklearn.model_selection import train_test_split


# ============================================================
# Save / load best hyperparameters
# ============================================================
def save_logging(dictionary, log_name):
    with open(log_name, "w") as f:
        json.dump(dictionary, f, indent=4)


def load_logging(filename):
    with open(filename, "r") as f:
        return json.load(f)


# ============================================================
# MLP random-search space
#
# Architecture candidates are taken from the original DeepHit
# MLP repository.
#
# Experimental/training hyperparameters are fixed to the same
# values used in the final DeepHit-KAN experiment.
# ============================================================
def get_random_hyperparameters(out_path, rng):

    # Fixed experimental protocol
    SET_BATCH_SIZE = [32]

    # Original DeepHit-MLP architecture search space
    SET_LAYERS = [1, 2, 3, 5]
    SET_NODES = [50, 100, 200, 300]
    SET_ACTIVATION_FN = ['relu', 'elu', 'tanh']

    return {
        # Training protocol
        'mb_size': rng.choice(SET_BATCH_SIZE),
        'iteration': 8000,
        'lr_train': 1e-4,
        'keep_prob': 0.6,

        # Shared MLP
        'h_dim_shared': rng.choice(SET_NODES),
        'num_layers_shared': rng.choice(SET_LAYERS),

        # Cause-specific MLPs
        'h_dim_CS': rng.choice(SET_NODES),
        'num_layers_CS': rng.choice(SET_LAYERS),

        # MLP non-linearity
        'active_fn': rng.choice(SET_ACTIVATION_FN),

        # DeepHit loss
        'alpha': 1.0,
        'beta': 1.0,
        'gamma': 0.0,

        'out_path': out_path
    }


# ============================================================
# Experimental settings
# SAME protocol as final DeepHit-KAN
# ============================================================
OUT_ITERATION = 5
RS_ITERATION = 8

data_mode = "SYNTHETIC"

if data_mode == "SYNTHETIC":
    (
        x_dim,
        (data, time, label),
        (mask1, mask2)
    ) = impt.import_dataset_SYNTHETIC()

    EVAL_TIMES = [12, 24, 36]

elif data_mode == "METABRIC":
    (
        x_dim,
        (data, time, label),
        (mask1, mask2)
    ) = impt.import_dataset_METABRIC()

    EVAL_TIMES = [144, 288, 432]

else:
    raise ValueError("Dataset not found.")


# IMPORTANT:
# Use a different output folder from DeepHit-KAN so that the
# MLP run cannot overwrite KAN checkpoints/results.
out_path = f"{data_mode}/results_MLP/"
os.makedirs(out_path, exist_ok=True)


# ============================================================
# Containers for OUTER-TEST results
# ============================================================
num_Event_total = mask1.shape[1]

event_c_index = np.full(
    (
        OUT_ITERATION,
        num_Event_total,
        len(EVAL_TIMES)
    ),
    np.nan
)

avg_c_index = np.full(
    (
        OUT_ITERATION,
        len(EVAL_TIMES)
    ),
    np.nan
)


# ============================================================
# Random-search RNG
# Same root seed as final DeepHit-KAN
# ============================================================
RS_RNG = random.Random(1234)


# ============================================================
# OUTER LOOP
# Each outer iteration:
#   1) makes the untouched 20% outer test split
#   2) runs 8 random-search configurations on development data
#   3) loads the winning checkpoint
#   4) evaluates it once on the untouched outer test set
# ============================================================
for out_itr in range(OUT_ITERATION):

    print(
        f"\n=== OUTER ITERATION {out_itr} "
        f"— RANDOM SEARCH ==="
    )

    best_valid = -999
    best_parser = None

    itr_path = f"{out_path}/itr_{out_itr}"
    os.makedirs(
        itr_path,
        exist_ok=True
    )

    # --------------------------------------------------------
    # OUTER DEVELOPMENT / TEST SPLIT
    # Same seed protocol as DeepHit-KAN
    # --------------------------------------------------------
    (
        tr_data,
        te_data,
        tr_time,
        te_time,
        tr_label,
        te_label,
        tr_mask1,
        te_mask1,
        tr_mask2,
        te_mask2
    ) = train_test_split(
        data,
        time,
        label,
        mask1,
        mask2,
        test_size=0.20,
        random_state=1234 + out_itr
    )

    # --------------------------------------------------------
    # Reconstruct the INNER-TRAIN indices used by get_main.
    #
    # These are needed only as the training reference sample
    # for the IPCW weighted C-index in the final OUTER test.
    #
    # This must reproduce exactly the inner split used in
    # get_valid_performance().
    # --------------------------------------------------------
    inner_indices = np.arange(
        len(tr_data)
    )

    metric_train_idx, _ = train_test_split(
        inner_indices,
        test_size=0.20,
        random_state=1234 + out_itr
    )

    metric_tr_time = tr_time[
        metric_train_idx
    ]

    metric_tr_label = tr_label[
        metric_train_idx
    ]

    development_data = (
        tr_data,
        tr_time,
        tr_label
    )

    development_mask = (
        tr_mask1,
        tr_mask2
    )

    # ========================================================
    # RANDOM SEARCH: 8 configurations
    # ========================================================
    for rs_itr in range(RS_ITERATION):

        print(
            f"  [RS {rs_itr + 1}/{RS_ITERATION}]"
        )

        # ----------------------------------------------------
        # 1) Sample MLP architecture
        # ----------------------------------------------------
        parser = get_random_hyperparameters(
            out_path,
            RS_RNG
        )

        print(parser)

        # ----------------------------------------------------
        # 2) Train + INNER validation
        # ----------------------------------------------------
        c_valid = get_valid_performance(
            development_data,
            development_mask,
            parser,
            out_itr,
            seed=1234 + out_itr,
            eval_time=EVAL_TIMES,
            MAX_VALUE=best_valid
        )

        # ----------------------------------------------------
        # 3) Keep best configuration for this outer iteration
        # ----------------------------------------------------
        if c_valid > best_valid:
            best_valid = c_valid
            best_parser = parser

            save_logging(
                best_parser,
                f"{itr_path}/hyperparameters_log.json"
            )

    print(
        f"  >> Mejor C-index de validación "
        f"en itr {out_itr}: {best_valid:.4f}"
    )

    if best_parser is None:
        raise RuntimeError(
            "No valid hyperparameter configuration "
            f"was found for outer iteration {out_itr}."
        )

    # ========================================================
    # OUTER-TEST EVALUATION
    # ========================================================
    print(
        f"\n=== OUTER ITERATION {out_itr} "
        f"— TEST EVALUATION ==="
    )

    input_dims = {
        'x_dim': x_dim,
        'num_Event': mask1.shape[1],
        'num_Category': mask1.shape[2]
    }

    # Architecture of the winning MLP
    network_settings = get_network_settings(
        best_parser,
        seed=1234 + out_itr
    )

    model = Model_DeepHit(
        input_dims,
        network_settings
    )

    model_path = (
        f"{itr_path}/models/"
        f"model_itr_{out_itr}.pt"
    )

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model checkpoint not found: {model_path}"
        )

    model.load_state_dict(
        torch.load(
            model_path,
            map_location='cpu'
        )
    )

    model.eval()

    # --------------------------------------------------------
    # Load EXACT normalizer associated with the winning model
    # --------------------------------------------------------
    normalizer_path = (
        f"{itr_path}/models/"
        f"normalizer_itr_{out_itr}.npz"
    )

    if not os.path.exists(normalizer_path):
        raise FileNotFoundError(
            f"Normalizer not found: {normalizer_path}"
        )

    normalizer_file = np.load(
        normalizer_path
    )

    normalizer = {
        'mode': str(normalizer_file['mode']),
        'offset': normalizer_file['offset'],
        'scale': normalizer_file['scale']
    }

    # Apply INNER-TRAIN-fitted normalizer to untouched test.
    te_data_normalized = impt.apply_normalizer(
        te_data,
        normalizer
    )

    te_x = torch.tensor(
        te_data_normalized,
        dtype=torch.float32
    )

    with torch.no_grad():
        pred = model.predict(
            te_x
        ).cpu().numpy()

    # ========================================================
    # Test weighted C-index
    # ========================================================
    for t_idx, t_horizon in enumerate(EVAL_TIMES):

        if t_horizon >= input_dims['num_Category']:
            raise ValueError(
                f"Evaluation horizon {t_horizon} "
                f"is outside num_Category="
                f"{input_dims['num_Category']}."
            )

        # Event-specific CIF/risk up to the horizon
        risk = np.sum(
            pred[:, :, :(t_horizon + 1)],
            axis=2
        )

        # ----------------------------------------------------
        # C-index per competing event
        # ----------------------------------------------------
        for e in range(
            input_dims['num_Event']
        ):

            ci = weighted_c_index(
                metric_tr_time[:, 0],
                (
                    metric_tr_label[:, 0]
                    == e + 1
                ).astype(int),
                risk[:, e],
                te_time[:, 0],
                (
                    te_label[:, 0]
                    == e + 1
                ).astype(int),
                t_horizon
            )

            if (
                not np.isfinite(ci)
                or ci < 0
            ):
                raise RuntimeError(
                    f"Invalid test C-index: "
                    f"outer={out_itr}, "
                    f"event={e + 1}, "
                    f"time={t_horizon}, "
                    f"value={ci}"
                )

            event_c_index[
                out_itr,
                e,
                t_idx
            ] = ci

        # Average across competing events
        avg_c_index[
            out_itr,
            t_idx
        ] = np.mean(
            event_c_index[
                out_itr,
                :,
                t_idx
            ]
        )

        print(
            f"  [Time {t_horizon}] "
            f"Average C-index: "
            f"{avg_c_index[out_itr, t_idx]:.4f}"
        )


# ============================================================
# FINAL SUMMARY ACROSS THE 5 OUTER ITERATIONS
# ============================================================
print(
    "\n\n=== FINAL SUMMARY ==="
)

for t_idx, t_horizon in enumerate(EVAL_TIMES):

    c_idx = avg_c_index[
        :,
        t_idx
    ]

    print(
        f"[Time {t_horizon}] "
        f"Mean C-index: {np.mean(c_idx):.4f} | "
        f"Std: {np.std(c_idx):.4f}"
    )


print(
    "\n=== EVENT-SPECIFIC RESULTS ==="
)

for e in range(num_Event_total):

    print(
        f"\nEvent {e + 1}"
    )

    for t_idx, t_horizon in enumerate(EVAL_TIMES):

        values = event_c_index[
            :,
            e,
            t_idx
        ]

        print(
            f"[Time {t_horizon}] "
            f"Mean C-index: {np.mean(values):.4f} | "
            f"Std: {np.std(values):.4f}"
        )
