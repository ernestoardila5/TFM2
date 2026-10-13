from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

import import_data as impt
from class_DeepHit import Model_DeepHit
from get_main_CTD_1L import get_network_settings


OUTER = 2
TARGET_NODE = 17
EVAL_TIMES = [144, 288, 432]

BASE = Path("METABRIC/results_KAN1L_CTD")
FEATURE_FILE = Path("./sample data/METABRIC/cleaned_features_final.csv")
OUTDIR = Path("KAN1L_CIF_node_sensitivity_outer2")


def load_normalizer(path):
    nf = np.load(path)
    return {
        "mode": str(nf["mode"]),
        "offset": nf["offset"],
        "scale": nf["scale"],
    }


def main():
    OUTDIR.mkdir(exist_ok=True)

    df = pd.read_csv(FEATURE_FILE)
    raw_data = df.to_numpy(dtype=float)

    x_dim, DATA, MASK = impt.import_dataset_METABRIC()
    mask1, _ = MASK

    all_idx = np.arange(len(raw_data))
    dev_idx, _ = train_test_split(
        all_idx,
        test_size=0.20,
        random_state=1234 + OUTER,
    )

    itr_dir = BASE / f"itr_{OUTER}"
    hp_path = itr_dir / "hyperparameters_log.json"
    model_path = itr_dir / "models" / f"model_itr_{OUTER}.pt"
    norm_path = itr_dir / "models" / f"normalizer_itr_{OUTER}.npz"

    for p in [hp_path, model_path, norm_path]:
        if not p.exists():
            raise FileNotFoundError(p)

    with hp_path.open("r", encoding="utf-8") as f:
        hp = json.load(f)

    normalizer = load_normalizer(norm_path)

    input_dims = {
        "x_dim": x_dim,
        "num_Event": mask1.shape[1],
        "num_Category": mask1.shape[2],
    }

    if input_dims["num_Event"] != 1:
        raise RuntimeError("This script assumes single-event METABRIC.")

    model = Model_DeepHit(
        input_dims,
        get_network_settings(hp, seed=1234 + OUTER),
    )
    model.load_state_dict(
        torch.load(model_path, map_location="cpu")
    )
    model.eval()

    Xn = impt.apply_normalizer(raw_data, normalizer)
    Xdev = torch.tensor(Xn[dev_idx], dtype=torch.float32)

    # --------------------------------------------------------
    # Obtain real CS representation c(x)
    # --------------------------------------------------------
    with torch.no_grad():
        shared = model.shared_kan(Xdev)
        q = torch.cat([Xdev, shared], dim=1)
        c = model.cs_kans[0](q)

    W = model.output_layer.weight.detach()
    b = model.output_layer.bias.detach()

    if W.shape[1] != c.shape[1]:
        raise RuntimeError("Linear-layer / CS-node dimension mismatch.")

    n_nodes = c.shape[1]
    node_sd = torch.std(c, dim=0, unbiased=False)

    all_tables = []
    validation_rows = []

    print("\n" + "=" * 100)
    print("CS NODE SENSITIVITY TO THE FINAL CIF — OUTER 2")
    print("=" * 100)
    print(f"Development patients: {len(Xdev)}")
    print(f"CS nodes: {n_nodes}")
    print(f"Discrete times: {W.shape[0]}")

    for horizon in EVAL_TIMES:
        if horizon >= W.shape[0]:
            print(f"\nTime {horizon}: OUTSIDE output dimension, skipped.")
            continue

        # ====================================================
        # Analytic derivative
        #
        # z_s = b_s + sum_j W_sj c_j
        # p_s = softmax(z)_s
        # F_h = sum_{tau<=h} p_tau
        #
        # dF_h/dc_j =
        #   sum_{tau<=h} p_tau W_tau,j
        #   - F_h * sum_s p_s W_s,j
        # ====================================================
        with torch.no_grad():
            logits = c @ W.T + b[None, :]
            p = torch.softmax(logits, dim=1)

            Fh = torch.sum(p[:, :horizon + 1], dim=1)

            first_term = p[:, :horizon + 1] @ W[:horizon + 1, :]
            expected_W = p @ W

            grad_analytic = (
                first_term
                - Fh[:, None] * expected_W
            )

        # ====================================================
        # Independent autograd validation of the derivative
        # ====================================================
        c_ag = c.detach().clone().requires_grad_(True)
        logits_ag = c_ag @ W.T + b[None, :]
        p_ag = torch.softmax(logits_ag, dim=1)
        Fh_ag = torch.sum(
            p_ag[:, :horizon + 1],
            dim=1
        )

        grad_autograd = torch.autograd.grad(
            Fh_ag.sum(),
            c_ag,
            retain_graph=False,
            create_graph=False,
        )[0].detach()

        grad_err = torch.abs(
            grad_analytic - grad_autograd
        )

        max_grad_err = float(grad_err.max())
        mean_grad_err = float(grad_err.mean())

        validation_rows.append({
            "time_index": horizon,
            "max_analytic_vs_autograd_error": max_grad_err,
            "mean_analytic_vs_autograd_error": mean_grad_err,
        })

        # ====================================================
        # Node ranking
        # ====================================================
        grad_rms = torch.sqrt(
            torch.mean(grad_analytic ** 2, dim=0)
        )

        grad_mean_abs = torch.mean(
            torch.abs(grad_analytic),
            dim=0
        )

        grad_mean = torch.mean(
            grad_analytic,
            dim=0
        )

        positive_fraction = torch.mean(
            (grad_analytic > 0).float(),
            dim=0
        )

        # Scale-comparable local sensitivity:
        # one observed SD change in node value, locally.
        standardized_rms = node_sd * grad_rms
        standardized_mean_abs = node_sd * grad_mean_abs

        table = pd.DataFrame({
            "time_index": horizon,
            "node": np.arange(n_nodes),
            "node_sd": node_sd.cpu().numpy(),
            "dCIF_dc_rms": grad_rms.cpu().numpy(),
            "dCIF_dc_mean_abs": grad_mean_abs.cpu().numpy(),
            "dCIF_dc_mean": grad_mean.cpu().numpy(),
            "positive_gradient_fraction": positive_fraction.cpu().numpy(),
            "standardized_sensitivity_rms": standardized_rms.cpu().numpy(),
            "standardized_sensitivity_mean_abs": standardized_mean_abs.cpu().numpy(),
        })

        table = table.sort_values(
            "standardized_sensitivity_rms",
            ascending=False,
        ).reset_index(drop=True)

        table.insert(
            1,
            "rank_standardized",
            np.arange(1, len(table) + 1),
        )

        all_tables.append(table)

        target = table[
            table["node"] == TARGET_NODE
        ].iloc[0]

        print(f"\n{'-' * 100}")
        print(f"CIF HORIZON t = {horizon}")
        print(f"{'-' * 100}")
        print(
            "Analytic derivative vs autograd: "
            f"max error={max_grad_err:.12e} | "
            f"mean error={mean_grad_err:.12e}"
        )

        print("\nTOP 10 CS NODES BY STANDARDIZED CIF SENSITIVITY")
        cols = [
            "rank_standardized",
            "node",
            "node_sd",
            "dCIF_dc_rms",
            "dCIF_dc_mean_abs",
            "standardized_sensitivity_rms",
            "positive_gradient_fraction",
        ]
        print(
            table.loc[:, cols]
            .head(10)
            .to_string(index=False)
        )

        print(f"\nNODE {TARGET_NODE}")
        print(
            f"rank={int(target['rank_standardized'])}/{n_nodes} | "
            f"node_SD={target['node_sd']:.10f} | "
            f"RMS(dF/dc17)={target['dCIF_dc_rms']:.10f} | "
            f"mean|dF/dc17|={target['dCIF_dc_mean_abs']:.10f} | "
            f"standardized_RMS="
            f"{target['standardized_sensitivity_rms']:.10f} | "
            f"positive_fraction="
            f"{target['positive_gradient_fraction']:.4f}"
        )

    if all_tables:
        pd.concat(
            all_tables,
            ignore_index=True,
        ).to_csv(
            OUTDIR / "cs_nodes_ranked_by_CIF_sensitivity.csv",
            index=False,
        )

    pd.DataFrame(validation_rows).to_csv(
        OUTDIR / "analytic_vs_autograd_validation.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Exact derivative formula documentation
    # --------------------------------------------------------
    text = r"""CS-NODE SENSITIVITY TO THE DEEPHIT CIF

For one-event DeepHit:

    z_s(x) = b_s + SUM_j W_(s,j) c_j(x)

    p_s(x) = exp(z_s) / SUM_r exp(z_r)

    F_h(x) = SUM_(tau <= h) p_tau(x)

The exact local derivative of the CIF with respect to CS node c_j is:

    dF_h/dc_j
      = SUM_(tau <= h) p_tau W_(tau,j)
        - F_h SUM_s p_s W_(s,j)

This is NOT the same as looking only at W_(h,j)c_j.
The CIF at horizon h depends on all probabilities up to h and,
through the softmax denominator, on all discrete-time logits.

For ranking nodes on a comparable observed scale, this script uses:

    standardized_sensitivity_rms(j,h)
      = SD_dev(c_j) *
        sqrt( mean_dev[ (dF_h/dc_j)^2 ] )

This remains a local model-sensitivity measure, not a causal effect.
"""
    (OUTDIR / "CIF_node_sensitivity_formula.txt").write_text(
        text,
        encoding="utf-8",
    )

    print("\nFILES")
    print(OUTDIR / "cs_nodes_ranked_by_CIF_sensitivity.csv")
    print(OUTDIR / "analytic_vs_autograd_validation.csv")
    print(OUTDIR / "CIF_node_sensitivity_formula.txt")
    print("=" * 100)


if __name__ == "__main__":
    main()
