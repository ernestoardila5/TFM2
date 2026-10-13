import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from import_data import import_dataset_SYNTHETIC
from class_DeepHit import Model_DeepHit

def main():
    # ---------------------------
    # 1) Cargar dataset
    # ---------------------------
    x_dim, DATA, MASK = import_dataset_SYNTHETIC(norm_mode="standard")
    data, time, label = DATA
    mask1, mask2 = MASK

    num_Event = mask1.shape[1]
    num_Category = mask1.shape[2]

    # ---------------------------
    # 2) Reconstruir el modelo (misma arquitectura que usaste)
    # ---------------------------
    input_dims = {
        "x_dim": x_dim,
        "num_Event": num_Event,
        "num_Category": num_Category
    }

    network_settings = {
        "h_dim_shared": 32,
        "num_layers_shared": 1,
        "h_dim_CS": 32,
        "num_layers_CS": 1
    }

    model = Model_DeepHit(input_dims, network_settings)

    # ---------------------------
    # 3) Cargar checkpoint
    # ---------------------------
    ckpt_path = "./debug/itr_0/models/model_itr_0.pt"
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"No encuentro el checkpoint: {ckpt_path}")

    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model.eval()

    # ---------------------------
    # 4) Preparar un batch representativo para "ver" activaciones
    #    (muy importante para plot/attribute)
    # ---------------------------
    # Elegimos un subconjunto aleatorio de datos (ej: 2048 muestras)
    rng = np.random.default_rng(123)
    idx = rng.choice(data.shape[0], size=2048, replace=False)
    x = torch.tensor(data[idx]).float()

    # Forward: esto hace que el shared_kan guarde cache_data + activaciones (acts)
    with torch.no_grad():
        _ = model.shared_kan(x)

    shared = model.shared_kan

    # ---------------------------
    # 5) (A) Importancia por feature (global)
    # ---------------------------
    # attribute() calcula node_scores y edge_scores
    shared.attribute()

    # node_scores[0] suele corresponder a importancia de inputs para outputs
    # (promediamos sobre neuronas de salida para obtener importancia global de cada feature)
    # Nota: si la forma exacta difiere, lo ajustamos, pero esto suele funcionar bien.
    node0 = shared.node_scores[0]   # shape: (n_features,)
    feat_importance = node0.detach().cpu().numpy()
    order = np.argsort(-feat_importance)

    os.makedirs("./figures/shared_kan", exist_ok=True)

    plt.figure(figsize=(10,4))
    plt.bar(range(len(feat_importance)), feat_importance[order])
    plt.xticks(range(len(feat_importance)), [f"X{j}" for j in order], rotation=45)
    plt.title("Shared KAN - importancia global por feature (node_scores)")
    plt.tight_layout()
    plt.savefig("./figures/shared_kan/feature_importance_shared.png", dpi=200)
    plt.close()

    print("✅ Guardada: ./figures/shared_kan/feature_importance_shared.png")
    print("Top features:", order[:5], "con importancias:", feat_importance[order[:5]])

    # ---------------------------
    # 6) (B) Plot de TODAS las splines del shared_kan
    # ---------------------------
    # OJO: para width=[12,32] esto genera 12*32 = 384 plots (no es un bug).
    # Es literalmente "todas las conexiones" input->neurona.
    shared.plot(folder="./figures/shared_kan/splines_all", metric="backward", tick=False)

    print("✅ Splines guardadas en: ./figures/shared_kan/splines_all/")
    print("   (espera ~384 imágenes si es una sola capa 12->32)")

if __name__ == "__main__":
    main()
