import os
import json
import numpy as np
import torch

from sklearn.model_selection import train_test_split

import import_data as impt
from class_DeepHit import Model_DeepHit
from utils_eval_CTD import weighted_c_index
from get_main_CTD_1L import get_network_settings

EVAL_TIMES = [144, 288, 432]

MODELS = {
    "KAN_1L": "METABRIC/results_KAN1L_CTD",
    "KAN_MULTI": "METABRIC/results_KAN_MULTI_CTD",
}

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def evaluate_model(name, base_path, x_dim, data, time, label, mask1, mask2):
    out_results = np.full((5, len(EVAL_TIMES)), np.nan)
    input_dims = {
        "x_dim": x_dim,
        "num_Event": mask1.shape[1],
        "num_Category": mask1.shape[2],
    }

    print()
    print("=" * 70)
    print(name)
    print("=" * 70)

    for out_itr in range(5):
        (
            tr_data, te_data,
            tr_time, te_time,
            tr_label, te_label,
            tr_mask1, te_mask1,
            tr_mask2, te_mask2
        ) = train_test_split(
            data, time, label, mask1, mask2,
            test_size=0.20,
            random_state=1234 + out_itr
        )

        inner_indices = np.arange(len(tr_data))
        metric_train_idx, _ = train_test_split(
            inner_indices,
            test_size=0.20,
            random_state=1234 + out_itr
        )

        metric_tr_time = tr_time[metric_train_idx]
        metric_tr_label = tr_label[metric_train_idx]

        itr_path = os.path.join(base_path, "itr_" + str(out_itr))
        parser_path = os.path.join(itr_path, "hyperparameters_log.json")
        model_path = os.path.join(itr_path, "models", "model_itr_" + str(out_itr) + ".pt")
        normalizer_path = os.path.join(itr_path, "models", "normalizer_itr_" + str(out_itr) + ".npz")

        for required in [parser_path, model_path, normalizer_path]:
            if not os.path.exists(required):
                raise FileNotFoundError("Missing required file: " + required)

        parser = load_json(parser_path)
        network_settings = get_network_settings(parser, seed=1234 + out_itr)
        model = Model_DeepHit(input_dims, network_settings)

        state = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state)
        model.eval()

        nf = np.load(normalizer_path)
        normalizer = {
            "mode": str(nf["mode"]),
            "offset": nf["offset"],
            "scale": nf["scale"],
        }

        te_data_norm = impt.apply_normalizer(te_data, normalizer)
        te_x = torch.tensor(te_data_norm, dtype=torch.float32)

        with torch.no_grad():
            pred = model(te_x).numpy()

        print()
        print("Outer", out_itr)

        for t_idx, t_horizon in enumerate(EVAL_TIMES):
            risk = np.sum(pred[:, :, :t_horizon + 1], axis=2)
            event_scores = []

            for e in range(input_dims["num_Event"]):
                ci = weighted_c_index(
                    metric_tr_time[:, 0],
                    (metric_tr_label[:, 0] == e + 1).astype(int),
                    risk[:, e],
                    te_time[:, 0],
                    (te_label[:, 0] == e + 1).astype(int),
                    t_horizon
                )
                event_scores.append(ci)

            score = float(np.mean(event_scores))
            out_results[out_itr, t_idx] = score
            print("  Time {}: {:.6f}".format(t_horizon, score))

    print()
    print("FINAL", name)

    for t_idx, t_horizon in enumerate(EVAL_TIMES):
        values = out_results[:, t_idx]
        print(
            "Time {}: mean={:.6f} | SD sample={:.6f} | values={}".format(
                t_horizon,
                np.mean(values),
                np.std(values, ddof=1),
                np.array2string(values, precision=6, separator=", ")
            )
        )

    return out_results

x_dim, DATA, MASK = impt.import_dataset_METABRIC()
data, time, label = DATA
mask1, mask2 = MASK

all_results = {}

for name, base_path in MODELS.items():
    all_results[name] = evaluate_model(
        name, base_path, x_dim, data, time, label, mask1, mask2
    )

print()
print("=" * 70)
print("COMPARISON")
print("=" * 70)

for t_idx, t_horizon in enumerate(EVAL_TIMES):
    print()
    print("TIME", t_horizon)
    for name in MODELS:
        vals = all_results[name][:, t_idx]
        print(
            "  {:10s}: {:.6f} +/- {:.6f}".format(
                name,
                np.mean(vals),
                np.std(vals, ddof=1)
            )
        )
