import json
from pathlib import Path

BASE = Path("METABRIC/results_KAN1L_CTD")

KEYS = [
    "mb_size",
    "iteration",
    "lr_train",
    "keep_prob",
    "num_layers_shared",
    "num_layers_CS",
    "sum_nodes_shared",
    "mult_nodes_shared",
    "sum_nodes_CS",
    "mult_nodes_CS",
    "grid",
    "k",
    "grid_eps",
    "noise_scale",
    "grid_range",
    "sparse_init",
    "mult_arity",
    "alpha",
    "beta",
    "gamma",
]

print("=== WINNING HYPERPARAMETERS: DEEPHIT-KAN 1L ===")

for out_itr in range(5):
    p = BASE / f"itr_{out_itr}" / "hyperparameters_log.json"
    print(f"\n--- OUTER {out_itr} ---")
    print("file:", p)

    if not p.exists():
        print("MISSING")
        continue

    with p.open("r", encoding="utf-8") as f:
        hp = json.load(f)

    for key in KEYS:
        print(f"{key}: {hp.get(key)}")
