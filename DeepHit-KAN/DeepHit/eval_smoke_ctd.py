import json
import numpy as np
import torch

from sklearn.model_selection import train_test_split

import import_data as impt
from class_DeepHit import Model_DeepHit
from get_main_CTD_1L import get_network_settings
from utils_eval_CTD import ctd_index

x_dim, (data, time, label), (mask1, mask2) = (
    impt.import_dataset_METABRIC()
)

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
    random_state=1234
)

with open(
    'METABRIC/results_KAN1L_CTD_SMOKE/itr_0/hyperparameters_log.json'
) as f:
    parser = json.load(f)

network_settings = get_network_settings(
    parser,
    seed=1234
)

input_dims = {
    'x_dim': x_dim,
    'num_Event': mask1.shape[1],
    'num_Category': mask1.shape[2]
}

model = Model_DeepHit(
    input_dims,
    network_settings
)

model.load_state_dict(
    torch.load(
        'METABRIC/results_KAN1L_CTD_SMOKE/itr_0/models/model_itr_0.pt',
        map_location='cpu'
    )
)

z = np.load(
    'METABRIC/results_KAN1L_CTD_SMOKE/itr_0/models/normalizer_itr_0.npz'
)

normalizer = {
    'mode': str(z['mode']),
    'offset': z['offset'],
    'scale': z['scale']
}

te_data = impt.apply_normalizer(
    te_data,
    normalizer
)

model.eval()

with torch.no_grad():
    pred = model(
        torch.tensor(
            te_data,
            dtype=torch.float32
        )
    ).numpy()

score = ctd_index(
    pred,
    te_time[:, 0],
    te_label[:, 0],
    event=1
)

print()
print('SMOKE TEST Ctd =', score)