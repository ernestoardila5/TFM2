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
SOURCE_NAMES = ["age_at_diagnosis", "NPI"]
EVAL_TIMES = [144, 288, 432]

BASE = Path("METABRIC/results_KAN1L_CTD")
FEATURE_FILE = Path("./sample data/METABRIC/cleaned_features_final.csv")
OUTDIR = Path("KAN1L_exact_path_decomposition_outer2")


def load_normalizer(path):
    nf = np.load(path)
    return {
        "mode": str(nf["mode"]),
        "offset": nf["offset"],
        "scale": nf["scale"],
    }


def rms(x):
    return torch.sqrt(torch.mean(x ** 2))


def main():
    OUTDIR.mkdir(exist_ok=True)

    df = pd.read_csv(FEATURE_FILE)
    feature_names = list(df.columns)
    raw_data = df.to_numpy(dtype=float)

    source_indices = []
    for name in SOURCE_NAMES:
        if name not in feature_names:
            raise ValueError(f"Feature not found: {name}")
        source_indices.append(feature_names.index(name))

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

    model = Model_DeepHit(
        input_dims,
        get_network_settings(hp, seed=1234 + OUTER),
    )
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    Xn = impt.apply_normalizer(raw_data, normalizer)
    Xdev = torch.tensor(
        Xn[dev_idx],
        dtype=torch.float32,
        requires_grad=True,
    )

    n = Xdev.shape[0]
    n_original = Xdev.shape[1]

    shared = model.shared_kan(Xdev)
    q = torch.cat([Xdev, shared], dim=1)
    c = model.cs_kans[0](q)

    n_shared = shared.shape[1]
    n_cs = c.shape[1]

    W = model.output_layer.weight
    b = model.output_layer.bias
    logits = c @ W.T + b[None, :]
    p = torch.softmax(logits, dim=1)

    nsources = len(source_indices)

    J_shared = torch.zeros(
        (n, n_shared, nsources),
        dtype=Xdev.dtype,
    )

    print("\nComputing d(shared_k)/d(source) ...")
    for k in range(n_shared):
        g = torch.autograd.grad(
            shared[:, k].sum(),
            Xdev,
            retain_graph=True,
            create_graph=False,
        )[0]
        for s, idx in enumerate(source_indices):
            J_shared[:, k, s] = g[:, idx].detach()

        if (k + 1) % 8 == 0 or k == n_shared - 1:
            print(f"  shared nodes: {k+1}/{n_shared}")

    J_cs_direct = torch.zeros(
        (n, n_cs, nsources),
        dtype=Xdev.dtype,
    )
    J_cs_shared = torch.zeros(
        (n, n_cs, n_shared),
        dtype=Xdev.dtype,
    )

    print("\nComputing d(c_j)/d(q) ...")
    for j in range(n_cs):
        gq = torch.autograd.grad(
            c[:, j].sum(),
            q,
            retain_graph=True,
            create_graph=False,
        )[0]

        for s, idx in enumerate(source_indices):
            J_cs_direct[:, j, s] = gq[:, idx].detach()

        J_cs_shared[:, j, :] = gq[:, n_original:].detach()

        if (j + 1) % 8 == 0 or j == n_cs - 1:
            print(f"  CS nodes: {j+1}/{n_cs}")

    summary_rows = []
    path_rows = []

    print("\n" + "=" * 110)
    print("EXACT CHAIN-RULE PATH DECOMPOSITION — OUTER 2")
    print("=" * 110)
    print(f"Patients: {n}")
    print(f"Shared nodes: {n_shared}")
    print(f"CS nodes: {n_cs}")
    print(f"Sources: {SOURCE_NAMES}")

    for h in EVAL_TIMES:
        Fh = p[:, :h + 1].sum(dim=1)

        dF_dc = (
            p[:, :h + 1] @ W[:h + 1, :]
            - Fh[:, None] * (p @ W)
        ).detach()

        full_grad_all = torch.autograd.grad(
            Fh.sum(),
            Xdev,
            retain_graph=True,
            create_graph=False,
        )[0].detach()

        for s, (source_name, source_idx) in enumerate(
            zip(SOURCE_NAMES, source_indices)
        ):
            scale = float(normalizer["scale"][source_idx])

            direct_components = (
                dF_dc * J_cs_direct[:, :, s]
            ) / scale

            indirect_components = (
                dF_dc[:, :, None]
                * J_cs_shared
                * J_shared[:, None, :, s]
            ) / scale

            total_from_paths = (
                direct_components.sum(dim=1)
                + indirect_components.sum(dim=(1, 2))
            )

            full_grad = full_grad_all[:, source_idx] / scale
            err = torch.abs(total_from_paths - full_grad)

            direct_total = direct_components.sum(dim=1)
            indirect_total = indirect_components.sum(dim=(1, 2))

            direct_rms = float(rms(direct_total))
            indirect_rms = float(rms(indirect_total))
            total_rms = float(rms(full_grad))

            print("\n" + "-" * 110)
            print(f"{source_name} | horizon {h}")
            print("-" * 110)
            print(
                "Exact decomposition vs full autograd: "
                f"max error={float(err.max()):.12e} | "
                f"mean error={float(err.mean()):.12e}"
            )
            print(
                f"RMS total derivative    = {total_rms:.10f}\n"
                f"RMS direct aggregate   = {direct_rms:.10f}\n"
                f"RMS shared aggregate   = {indirect_rms:.10f}"
            )

            summary_rows.append({
                "time_index": h,
                "source_feature": source_name,
                "source_index": source_idx,
                "RMS_total_dF_dx": total_rms,
                "RMS_direct_aggregate": direct_rms,
                "RMS_shared_aggregate": indirect_rms,
                "mean_total_dF_dx": float(torch.mean(full_grad)),
                "positive_fraction_total": float(
                    torch.mean((full_grad > 0).float())
                ),
                "max_decomposition_error": float(err.max()),
                "mean_decomposition_error": float(err.mean()),
            })

            direct_scores = torch.sqrt(
                torch.mean(direct_components ** 2, dim=0)
            )

            top_direct = torch.argsort(
                direct_scores,
                descending=True,
            )[:10]

            print("\nTOP 10 DIRECT PATHS: source -> CS node -> CIF")
            for rank, j_t in enumerate(top_direct, start=1):
                j = int(j_t)
                comp = direct_components[:, j]
                score = float(direct_scores[j])

                print(
                    f"{rank:2d}. {source_name} -> c{j} -> F({h}) | "
                    f"RMS={score:.10f} | "
                    f"mean={float(comp.mean()):+.10f} | "
                    f"positive_fraction="
                    f"{float(torch.mean((comp > 0).float())):.4f}"
                )

                path_rows.append({
                    "time_index": h,
                    "source_feature": source_name,
                    "path_type": "direct",
                    "shared_node": np.nan,
                    "cs_node": j,
                    "RMS_path_contribution": score,
                    "mean_path_contribution": float(comp.mean()),
                    "positive_fraction": float(
                        torch.mean((comp > 0).float())
                    ),
                })

            indirect_scores = torch.sqrt(
                torch.mean(indirect_components ** 2, dim=0)
            )

            flat = indirect_scores.flatten()
            top_indirect_flat = torch.argsort(
                flat,
                descending=True,
            )[:15]

            print(
                "\nTOP 15 SHARED PATHS: "
                "source -> shared_k -> CS node -> CIF"
            )

            for rank, flat_t in enumerate(
                top_indirect_flat,
                start=1,
            ):
                flat_idx = int(flat_t)
                j = flat_idx // n_shared
                k = flat_idx % n_shared

                comp = indirect_components[:, j, k]
                score = float(indirect_scores[j, k])

                print(
                    f"{rank:2d}. {source_name} -> s{k} -> c{j} -> F({h}) | "
                    f"RMS={score:.10f} | "
                    f"mean={float(comp.mean()):+.10f} | "
                    f"positive_fraction="
                    f"{float(torch.mean((comp > 0).float())):.4f}"
                )

                path_rows.append({
                    "time_index": h,
                    "source_feature": source_name,
                    "path_type": "via_shared",
                    "shared_node": k,
                    "cs_node": j,
                    "RMS_path_contribution": score,
                    "mean_path_contribution": float(comp.mean()),
                    "positive_fraction": float(
                        torch.mean((comp > 0).float())
                    ),
                })

            grouped_shared = indirect_components.sum(dim=1)
            grouped_shared_scores = torch.sqrt(
                torch.mean(grouped_shared ** 2, dim=0)
            )
            top_grouped_shared = torch.argsort(
                grouped_shared_scores,
                descending=True,
            )[:10]

            print(
                "\nTOP 10 SHARED NODES AFTER SUMMING THEIR 32 CS ROUTES"
            )
            for rank, k_t in enumerate(
                top_grouped_shared,
                start=1,
            ):
                k = int(k_t)
                comp = grouped_shared[:, k]
                print(
                    f"{rank:2d}. {source_name} -> s{k} -> ALL CS -> F({h}) | "
                    f"RMS={float(grouped_shared_scores[k]):.10f} | "
                    f"mean={float(comp.mean()):+.10f}"
                )

    summary_df = pd.DataFrame(summary_rows)
    paths_df = pd.DataFrame(path_rows)

    summary_df.to_csv(
        OUTDIR / "age_NPI_total_decomposition_summary.csv",
        index=False,
    )

    paths_df.to_csv(
        OUTDIR / "age_NPI_ranked_path_components.csv",
        index=False,
    )

    print("\n" + "=" * 110)
    print("FILES")
    print(OUTDIR / "age_NPI_total_decomposition_summary.csv")
    print(OUTDIR / "age_NPI_ranked_path_components.csv")
    print("=" * 110)


if __name__ == "__main__":
    main()
