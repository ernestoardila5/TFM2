
import os
import torch
import random
import numpy as np
import import_data as impt
from class_DeepHit import Model_DeepHit
from utils_eval import weighted_c_index
from get_main import get_valid_performance, get_network_settings
import json
from sklearn.model_selection import train_test_split

#################save and load hyperparameters

def save_logging(dictionary, log_name):
    with open(log_name, "w") as f:
        json.dump(dictionary, f, indent=4)


def load_logging(filename):
    with open(filename, "r") as f:
        return json.load(f)

################################random hyperaparameter selection

def get_random_hyperparameters(out_path, rng):
    SET_BATCH_SIZE    = [32]#, 64, 128]
    SET_LAYERS        = [1]#, 3, 5] ##meter menos capas y menos neuronas
    # SET_NODES         = [10,20,30]
    # SET_BETA          = [0.1, 0.5, 1.0, 3.0, 5.0]
    SET_SUM_NODES_SHARED = [8, 16, 32, 64]
    SET_MULT_NODES_SHARED = [0, 4, 8]

    SET_SUM_NODES_CS = [8, 16, 32, 64]
    SET_MULT_NODES_CS = [0, 4, 8]
    SET_GRID = [5, 10, 15, 20, 30]
    SET_K = [1, 3]
    SET_GRID_EPS = [0.0, 0.02, 0.1, 0.5, 1.0]
    SET_NOISE = [0.1, 0.3, 0.5]
    SET_GRID_RANGE = [
        [-1, 1],
        [-2, 2],
        [-3, 3]
    ]
    SET_SPARSE_INIT = [False, True]

    return {
        'mb_size': rng.choice(SET_BATCH_SIZE),

        'iteration': 8000,
        'lr_train': 1e-4,
        'keep_prob': 0.6,

        'num_layers_shared': rng.choice(SET_LAYERS),
        'num_layers_CS': rng.choice(SET_LAYERS),

        'sum_nodes_shared': rng.choice(SET_SUM_NODES_SHARED),
        'mult_nodes_shared': rng.choice(SET_MULT_NODES_SHARED),

        'sum_nodes_CS': rng.choice(SET_SUM_NODES_CS),
        'mult_nodes_CS': rng.choice(SET_MULT_NODES_CS),

        'grid': rng.choice(SET_GRID),
        'k': rng.choice(SET_K),
        'grid_eps': rng.choice(SET_GRID_EPS),
        'noise_scale': rng.choice(SET_NOISE),
        'grid_range': rng.choice(SET_GRID_RANGE),
        'sparse_init': rng.choice(SET_SPARSE_INIT),

        'mult_arity': 2,

        'alpha': 1.0,
        'beta': 1.0,
        'gamma': 0.0,

        'out_path': out_path
    }



# ------------------------------
# Settings
# ------------------------------
OUT_ITERATION = 5  ##comprobar qué era lo de out iteration y rs iteration 
RS_ITERATION  = 8
data_mode = "SYNTHETIC"

if data_mode == "SYNTHETIC":
    (x_dim), (data, time, label), (mask1, mask2) = impt.import_dataset_SYNTHETIC()
    EVAL_TIMES = [12, 24, 36]
elif data_mode == "METABRIC":
    (x_dim), (data, time, label), (mask1, mask2) = impt.import_dataset_METABRIC()
    EVAL_TIMES = [144, 288, 432]
else:
    raise ValueError("Dataset not found.")

DATA = (data, time, label)
MASK = (mask1, mask2)

out_path = f"{data_mode}/results/"
os.makedirs(out_path, exist_ok=True)

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

RS_RNG = random.Random(1234)

# ------------------------------
# Random Search (entrenamiento + validación)
# ------------------------------
for out_itr in range(OUT_ITERATION):
    print(f"\n=== OUTER ITERATION {out_itr} — RANDOM SEARCH ===")

    best_valid = -999
    best_parser = None

    # carpeta para esta iteración externa
    itr_path = f"{out_path}/itr_{out_itr}"
    os.makedirs(itr_path, exist_ok=True)

    # Use 20% test set
    
    tr_data, te_data, tr_time, te_time, tr_label, te_label, tr_mask1, te_mask1, tr_mask2, te_mask2 = train_test_split(
        data, time, label, mask1, mask2, test_size=0.20, random_state=1234 + out_itr
    )

    # --------------------------------------------------
    # Reconstruir INNER TRAIN para las métricas IPCW
    # Debe ser el mismo split utilizado en get_main.py
    # --------------------------------------------------
    inner_indices = np.arange(len(tr_data))

    metric_train_idx, _ = train_test_split(
        inner_indices,
        test_size=0.20,
        random_state=1234 + out_itr
    )

    metric_tr_time = tr_time[metric_train_idx]
    metric_tr_label = tr_label[metric_train_idx]

    development_data = (tr_data, tr_time, tr_label)
    development_mask = (tr_mask1, tr_mask2)

    for rs_itr in range(RS_ITERATION):
        print(f"  [RS {rs_itr+1}/{RS_ITERATION}]")

        # 1) Elegir hiperparámetros aleatorios
        parser = get_random_hyperparameters(
            out_path,
            RS_RNG
        )

        # 2) Entrenar + validar con esos hiperparámetros
        c_valid = get_valid_performance(
            development_data,
            development_mask,
            parser,
            out_itr,
            seed=1234 + out_itr,
            eval_time=EVAL_TIMES,
            MAX_VALUE=best_valid,
            # OUT_ITERATION=OUT_ITERATION,
        )

        # 3) Actualizar el mejor
        if c_valid > best_valid:
            best_valid = c_valid
            best_parser = parser
            # Guardamos los hiperparámetros ganadores
            save_logging(best_parser, f"{itr_path}/hyperparameters_log.json")

    print(f"  >> Mejor C-index de validación en itr {out_itr}: {best_valid:.4f}")


# ------------------------------
# Run Evaluation
# ------------------------------


# for out_itr in range(OUT_ITERATION):

#     print(f"\n=== OUTER ITERATION {out_itr} ===")

#     # Load best hyperparameters
#     log_file = f"{data_mode}/results/itr_{out_itr}/hyperparameters_log.json"
#     if not os.path.exists(log_file):
#         print("  Skipping: no hyperparameters found.")
#         continue

    # parser = load_logging(log_file)
    input_dims = {
        'x_dim': x_dim,
        'num_Event': np.shape(mask1)[1],
        'num_Category': np.shape(mask1)[2]
    }

    if best_parser is None:
        raise RuntimeError(
            f"No valid hyperparameter configuration "
            f"was found for outer iteration {out_itr}."
        )

    # 3. Extraer SOLO los hiperparámetros de arquitectura
    network_settings = get_network_settings(best_parser, seed= 1234 + out_itr)

    # 4. Construir el modelo correctamente
    model = Model_DeepHit(input_dims, network_settings)
    model_path = f"{data_mode}/results/itr_{out_itr}/models/model_itr_{out_itr}.pt"

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model checkpoint not found: {model_path}"
        )

    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    normalizer_path = (
        f"{data_mode}/results/"
        f"itr_{out_itr}/models/"
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


    te_data_normalized = impt.apply_normalizer(
        te_data,
        normalizer
    )



    te_x = torch.tensor(
        te_data_normalized,
        dtype=torch.float32
    )

    with torch.no_grad():
        pred = model(te_x).numpy()

    for t_idx, t_horizon in enumerate(EVAL_TIMES):

        if t_horizon >= input_dims['num_Category']:
            raise ValueError(
                f"Evaluation horizon {t_horizon} "
                f"is outside num_Category="
                f"{input_dims['num_Category']}."
            )

        risk = np.sum(
            pred[:, :, :t_horizon + 1],
            axis=2
        )

        # --------------------------------------------------
        # C-index por evento
        # --------------------------------------------------
        for e in range(input_dims['num_Event']):

            ci = weighted_c_index(
                metric_tr_time[:, 0],
                (metric_tr_label[:, 0] == e + 1).astype(int),
                risk[:, e],
                te_time[:, 0],
                (te_label[:, 0] == e + 1).astype(int),
                t_horizon
            )

            if not np.isfinite(ci) or ci < 0:
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

        # --------------------------------------------------
        # Promedio de los eventos para este horizonte
        # ESTO VA FUERA DEL for e
        # --------------------------------------------------
        avg_c_index[out_itr, t_idx] = np.mean(
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

# ------------------------------
# Final Summary
# ------------------------------
print("\n\n=== FINAL SUMMARY ===")

for t_idx, t_horizon in enumerate(EVAL_TIMES):
    c_idx = avg_c_index[:, t_idx]

    print(
        f"[Time {t_horizon}] "
        f"Mean C-index: {np.mean(c_idx):.4f} | "
        f"Std: {np.std(c_idx):.4f}"
    )

print("\n=== EVENT-SPECIFIC RESULTS ===")

for e in range(num_Event_total):
    print(f"\nEvent {e + 1}")

    for t_idx, t_horizon in enumerate(EVAL_TIMES):
        values = event_c_index[:, e, t_idx]

        print(
            f"[Time {t_horizon}] "
            f"Mean C-index: {np.mean(values):.4f} | "
            f"Std: {np.std(values):.4f}"
        )