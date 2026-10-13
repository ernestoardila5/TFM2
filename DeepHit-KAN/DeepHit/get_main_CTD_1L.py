from class_DeepHit import Model_DeepHit
import torch
import os
import numpy as np 
from sklearn.model_selection import train_test_split
from utils_eval_CTD import ctd_index
from import_data import fit_normalizer, apply_normalizer

# Si no lo tienes aún, define esta función aquí
def f_get_minibatch(mb_size, data, label, time, mask1, mask2): ##creamos el minibatch para el entrenamiento
    idx = np.random.choice(len(data), mb_size, replace=False)
    return data[idx], label[idx], time[idx], mask1[idx], mask2[idx]

def get_network_settings(parser, seed=1):
    return {
        # Arquitectura Shared
        'num_layers_shared': parser['num_layers_shared'],
        'sum_nodes_shared': parser['sum_nodes_shared'],
        'mult_nodes_shared': parser['mult_nodes_shared'],

        # Arquitectura Cause-Specific
        'num_layers_CS': parser['num_layers_CS'],
        'sum_nodes_CS': parser['sum_nodes_CS'],
        'mult_nodes_CS': parser['mult_nodes_CS'],

        # Hiperparámetros KAN
        'grid': parser['grid'],
        'k': parser['k'],
        'grid_eps': parser['grid_eps'],
        'noise_scale': parser['noise_scale'],
        'grid_range': parser['grid_range'],
        'sparse_init': parser['sparse_init'],
        'mult_arity': parser['mult_arity'],
        'keep_prob': parser['keep_prob'],

        # Reproducibilidad
        'seed': seed,
    }

def get_valid_performance(DATA, MASK, in_parser, out_itr, seed, eval_time=None, MAX_VALUE=-99):

    np.random.seed(seed)
    torch.manual_seed(seed)

    (data, time, label) = DATA
    (mask1, mask2) = MASK

    x_dim = data.shape[1]
    _, num_Event, num_Category = mask1.shape

    mb_size = in_parser['mb_size'] #tamaño del batch
    iteration = in_parser['iteration'] #nº de pasos del entrenamiento
    lr_train = in_parser['lr_train'] #learning rate (tamaño de pasos)
    alpha = in_parser['alpha'] 
    beta = in_parser['beta']
    gamma = in_parser['gamma']

    input_dims = {
        'x_dim': x_dim,
        'num_Event': num_Event,
        'num_Category': num_Category
    }

    network_settings = get_network_settings(in_parser, seed=seed)

    file_path_final = in_parser['out_path'] + '/itr_' + str(out_itr)
    os.makedirs(file_path_final + '/models/', exist_ok=True)

    print(f'{file_path_final} (a:{alpha} b:{beta} c:{gamma})')

    # Modelo + Optimizer
    model = Model_DeepHit(input_dims, network_settings)
    #a partir de aquí tenemos una red con salida ypred= [B, num_event, num_category] PERO SIN ENTRENAR!!
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_train)

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

    # ============================================================
    # NORMALIZACIÓN
    # Fit SOLO con INNER TRAIN
    # ============================================================

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
    
    

    if eval_time is None:
        eval_time = [int(np.percentile(tr_time, 25)),
                     int(np.percentile(tr_time, 50)),
                     int(np.percentile(tr_time, 75))]

    max_valid = -99
    stop_flag = 0
    avg_loss = 0

    print("MAIN TRAINING ...")
    print("VALIDATION METRIC: Antolini Ctd")

    GRID_UPDATE_NUM = 10
    GRID_UPDATE_STOP = iteration // 2

    grid_update_freq = max(
        1,
        GRID_UPDATE_STOP // GRID_UPDATE_NUM
    )

    for itr in range(iteration): #aquí se define el bucle del entrenamiento


        if stop_flag >= 5:
            break

        x_mb, k_mb, t_mb, m1_mb, m2_mb = f_get_minibatch(mb_size, tr_data, tr_label, tr_time, tr_mask1, tr_mask2)

        # convertir a tensor
        x_mb = torch.tensor(x_mb, dtype=torch.float32)
        k_mb = torch.tensor(k_mb, dtype=torch.float32)
        t_mb = torch.tensor(t_mb, dtype=torch.float32)
        m1_mb = torch.tensor(m1_mb, dtype=torch.float32)
        m2_mb = torch.tensor(m2_mb, dtype=torch.float32)

        if (
            itr < GRID_UPDATE_STOP
            and itr % grid_update_freq == 0
        ):

            grid_batch_size = min(
                256,
                len(tr_data)
            )

            grid_idx = np.random.choice(
                len(tr_data),
                grid_batch_size,
                replace=False
            )

            grid_x = torch.tensor(
                tr_data[grid_idx],
                dtype=torch.float32
            )

            model.update_grids(grid_x)

        loss_curr = model.train_step(x_mb, t_mb, k_mb, m1_mb, m2_mb, alpha, beta, gamma, optimizer)
        avg_loss += loss_curr 

        if (itr + 1) % 200 == 0:
            avg_loss /= 200
            print(f'|| ITR: {itr + 1:04d} | Loss: {avg_loss:.4f}')
            avg_loss = 0

            with torch.no_grad():
                va_data_tensor = torch.tensor(va_data, dtype=torch.float32)
                pred = model.predict(va_data_tensor).numpy()

                va_ctd = np.zeros(num_Event, dtype=float)
                for k in range(num_Event):
                    ci = ctd_index(
                        pred,
                        va_time[:, 0],
                        va_label[:, 0],
                        event=k + 1
                    )
                    if not np.isfinite(ci) or ci < 0:
                        raise RuntimeError(
                            f"Invalid validation Ctd: event={k + 1}, value={ci}"
                        )
                    va_ctd[k] = ci
                tmp_valid = float(np.mean(va_ctd))

                if tmp_valid > max_valid:
                    stop_flag = 0
                    max_valid = tmp_valid
                    print(f'updated.... Ctd = {tmp_valid:.4f}')
                    if max_valid > MAX_VALUE:
                        model_path = (
                            file_path_final
                            + f'/models/model_itr_{out_itr}.pt'
                        )

                        scaler_path = (
                            file_path_final
                            + f'/models/normalizer_itr_{out_itr}.npz'
                        )

                        # Guardar modelo
                        torch.save(
                            model.state_dict(),
                            model_path
                        )

                        # Guardar normalización asociada EXACTAMENTE
                        # a ese checkpoint
                        np.savez(
                            scaler_path,
                            mode=normalizer['mode'],
                            offset=normalizer['offset'],
                            scale=normalizer['scale']
                        )
                else:
                    stop_flag += 1

    return max_valid

# # ============================================================
# # SANITY CHECK MAIN
# # ============================================================
# if __name__ == "__main__":

#     print("\n==============================")
#     print(" Running DeepHit + KAN sanity check ")
#     print("==============================\n")

#     # --------------------------------------------------------
#     # 1. Import dataset (usar SYNTHETIC para depuración)
#     # --------------------------------------------------------
#     from import_data import import_dataset_SYNTHETIC

#     x_dim, DATA, MASK = import_dataset_SYNTHETIC(norm_mode="standard")
#     data, time, label = DATA
#     mask1, mask2 = MASK

#     print("Dataset loaded")
#     print(f"  data   shape: {data.shape}")
#     print(f"  time   shape: {time.shape}")
#     print(f"  label  shape: {label.shape}")
#     print(f"  mask1  shape: {mask1.shape}")
#     print(f"  mask2  shape: {mask2.shape}\n")

#     # --------------------------------------------------------
#     # 2. Definir hiperparámetros mínimos (SIN random search)
#     # --------------------------------------------------------
#     parser = {
#         # Entrenamiento
#         'mb_size': 64,
#         'iteration': 5000,
#         'lr_train': 1e-3,

#         # DeepHit loss
#         'alpha': 1.0,
#         'beta': 1.0,
#         'gamma': 1.0,

#         # Shared KAN
#         'num_layers_shared': 1,
#         'sum_nodes_shared': 32,
#         'mult_nodes_shared': 4,

#         # Cause-specific KAN
#         'num_layers_CS': 1,
#         'sum_nodes_CS': 32,
#         'mult_nodes_CS': 4,

#         # KAN
#         'grid': 5,
#         'k': 3,
#         'grid_eps': 0.02,
#         'noise_scale': 0.3,
#         'grid_range': [-1, 1],
#         'sparse_init': False,
#         'mult_arity': 2,

#         'out_path': './debug'
#     }

#     print("Sanity-check hyperparameters:")
#     for k, v in parser.items():
#         print(f"  {k}: {v}")
#     print("")

#     # --------------------------------------------------------
#     # 3. Ejecutar entrenamiento + validación mínima
#     # --------------------------------------------------------
#     print("Starting get_valid_performance()...\n")

#     try:
#         from get_main import get_valid_performance

#         max_valid = get_valid_performance(
#             DATA=DATA,
#             MASK=MASK,
#             in_parser=parser,
#             out_itr=0,
#             eval_time=None,     # usa percentiles internos
#             MAX_VALUE=-99,
#             seed=1234
#         )

#         print("\n==============================")
#         print(" Sanity check FINISHED ")
#         print("==============================")
#         print(f"Returned validation score: {max_valid}\n")

#     except Exception as e:
#         print("\n==============================")
#         print(" ❌ SANITY CHECK FAILED ")
#         print("==============================")
#         print("Error:")
#         raise e
