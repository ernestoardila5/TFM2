from class_DeepHit_MLP_FINAL import Model_DeepHit

import os
import numpy as np
import torch

from sklearn.model_selection import train_test_split

from utils_eval_MLP_FINAL import weighted_c_index, ctd_index
from import_data_MLP_FINAL import fit_normalizer, apply_normalizer


# ============================================================
# Minibatch sampling
# Same protocol as the final DeepHit-KAN implementation
# ============================================================
def f_get_minibatch(mb_size, data, label, time, mask1, mask2):
    idx = np.random.choice(
        len(data),
        mb_size,
        replace=False
    )

    return (
        data[idx],
        label[idx],
        time[idx],
        mask1[idx],
        mask2[idx]
    )


# ============================================================
# MLP network settings
# Only architecture-specific part changed from KAN -> MLP
# ============================================================
def get_network_settings(parser, seed=1):
    return {
        # Shared MLP
        'h_dim_shared': parser['h_dim_shared'],
        'num_layers_shared': parser['num_layers_shared'],

        # Cause-specific MLPs
        'h_dim_CS': parser['h_dim_CS'],
        'num_layers_CS': parser['num_layers_CS'],

        # MLP activation
        'active_fn': parser['active_fn'],

        # Same dropout protocol
        'keep_prob': parser['keep_prob'],

        # Reproducibility / compatibility with common pipeline
        'seed': seed,
    }


# ============================================================
# Train one random-search configuration and return its
# best INNER-validation performance.
#
# Experimental protocol is intentionally kept equal to the
# final DeepHit-KAN pipeline:
#   - same seed handling
#   - same inner split
#   - normalization fitted ONLY on inner train
#   - same optimizer/loss parameters received from parser
#   - validation every 200 iterations
#   - same early-stopping rule
#   - same weighted C-index
#   - same checkpoint/scaler saving logic
#
# The only model-specific change is KAN -> MLP.
# ============================================================
def get_valid_performance(
    DATA,
    MASK,
    in_parser,
    out_itr,
    seed,
    eval_time=None,
    MAX_VALUE=-99
):
    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------
    np.random.seed(seed)
    torch.manual_seed(seed)

    # --------------------------------------------------------
    # Data
    # IMPORTANT:
    # DATA here must already correspond to the OUTER-development
    # set selected by main_RandomSearch.
    # The outer test set must never enter this function.
    # --------------------------------------------------------
    data, time, label = DATA
    mask1, mask2 = MASK

    x_dim = data.shape[1]
    _, num_Event, num_Category = mask1.shape

    # --------------------------------------------------------
    # Training hyperparameters
    # --------------------------------------------------------
    mb_size = in_parser['mb_size']
    iteration = in_parser['iteration']
    lr_train = in_parser['lr_train']

    alpha = in_parser['alpha']
    beta = in_parser['beta']
    gamma = in_parser['gamma']

    input_dims = {
        'x_dim': x_dim,
        'num_Event': num_Event,
        'num_Category': num_Category
    }

    network_settings = get_network_settings(
        in_parser,
        seed=seed
    )

    # --------------------------------------------------------
    # Output paths
    # --------------------------------------------------------
    file_path_final = (
        in_parser['out_path']
        + '/itr_'
        + str(out_itr)
    )

    os.makedirs(
        file_path_final + '/models/',
        exist_ok=True
    )

    print(
        f'{file_path_final} '
        f'(a:{alpha} b:{beta} c:{gamma})'
    )

    # --------------------------------------------------------
    # Model + optimizer
    # Same optimizer family as DeepHit-KAN
    # --------------------------------------------------------
    model = Model_DeepHit(
        input_dims,
        network_settings
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr_train
    )

    # --------------------------------------------------------
    # INNER TRAIN / VALIDATION SPLIT
    # Same split protocol as DeepHit-KAN
    # --------------------------------------------------------
    (
        tr_data,
        va_data,
        tr_time,
        va_time,
        tr_label,
        va_label,
        tr_mask1,
        va_mask1,
        tr_mask2,
        va_mask2
    ) = train_test_split(
        data,
        time,
        label,
        mask1,
        mask2,
        test_size=0.20,
        random_state=seed
    )

    # --------------------------------------------------------
    # NORMALIZATION
    # Fit ONLY on inner train, then apply to train + validation.
    # This prevents validation leakage.
    # --------------------------------------------------------
    normalizer = fit_normalizer(
        tr_data,
        norm_mode='standard'
    )

    tr_data = apply_normalizer(
        tr_data,
        normalizer
    )

    va_data = apply_normalizer(
        va_data,
        normalizer
    )

    # --------------------------------------------------------
    # Evaluation horizons
    # --------------------------------------------------------

    max_valid = -99
    stop_flag = 0
    avg_loss = 0.0

    print("MAIN TRAINING ...")
    print("VALIDATION METRIC: Antolini C^td")

    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------
    for itr in range(iteration):

        # Same early-stopping rule as final KAN pipeline:
        # five consecutive validation checks without improvement.
        if stop_flag >= 5:
            break

        (
            x_mb,
            k_mb,
            t_mb,
            m1_mb,
            m2_mb
        ) = f_get_minibatch(
            mb_size,
            tr_data,
            tr_label,
            tr_time,
            tr_mask1,
            tr_mask2
        )

        # NumPy -> Torch
        x_mb = torch.tensor(
            x_mb,
            dtype=torch.float32
        )

        k_mb = torch.tensor(
            k_mb,
            dtype=torch.float32
        )

        t_mb = torch.tensor(
            t_mb,
            dtype=torch.float32
        )

        m1_mb = torch.tensor(
            m1_mb,
            dtype=torch.float32
        )

        m2_mb = torch.tensor(
            m2_mb,
            dtype=torch.float32
        )

        # ----------------------------------------------------
        # One MLP optimization step
        # No KAN grid update exists here.
        # ----------------------------------------------------
        loss_curr = model.train_step(
            x_mb,
            t_mb,
            k_mb,
            m1_mb,
            m2_mb,
            alpha,
            beta,
            gamma,
            optimizer
        )

        avg_loss += loss_curr

        # ----------------------------------------------------
        # VALIDATION
        # Same cadence as final DeepHit-KAN: every 200 steps
        # ----------------------------------------------------
        if (itr + 1) % 200 == 0:
            avg_loss /= 200

            print(
                f'|| ITR: {itr + 1:04d} '
                f'| Loss: {avg_loss:.4f}'
            )

            avg_loss = 0.0

            va_data_tensor = torch.tensor(
                va_data,
                dtype=torch.float32
            )

            pred = model.predict(
                va_data_tensor
            ).cpu().numpy()

                        # =================================================
            # Antolini C^td on INNER VALIDATION
            # =================================================
            va_ctd = np.zeros(
                num_Event,
                dtype=float
            )

            for k in range(num_Event):

                ci = ctd_index(
                    pred,
                    va_time[:, 0],
                    va_label[:, 0],
                    event=k + 1
                )

                if (
                    not np.isfinite(ci)
                    or ci < 0
                ):
                    raise RuntimeError(
                        f"Invalid validation C^td: "
                        f"event={k + 1}, "
                        f"value={ci}"
                    )

                va_ctd[k] = ci

            # METABRIC has a single event.
            # This remains generic for multiple events.
            tmp_valid = np.mean(
                va_ctd
            )

            # ------------------------------------------------
            # Local best for this random-search configuration
            # ------------------------------------------------
            if tmp_valid > max_valid:
                stop_flag = 0
                max_valid = tmp_valid

                print(
                    'updated.... C^td = '
                    f'{tmp_valid:.4f}'
                )

                # --------------------------------------------
                # Save only if this configuration also beats
                # the best random-search configuration already
                # found for this outer iteration.
                # --------------------------------------------
                if max_valid > MAX_VALUE:
                    model_path = (
                        file_path_final
                        + f'/models/model_itr_{out_itr}.pt'
                    )

                    scaler_path = (
                        file_path_final
                        + f'/models/normalizer_itr_{out_itr}.npz'
                    )

                    # Model checkpoint
                    torch.save(
                        model.state_dict(),
                        model_path
                    )

                    # EXACT normalizer associated with checkpoint
                    np.savez(
                        scaler_path,
                        mode=normalizer['mode'],
                        offset=normalizer['offset'],
                        scale=normalizer['scale']
                    )

            else:
                stop_flag += 1

    return max_valid
