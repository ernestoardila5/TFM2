from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from scipy.interpolate import BSpline, PPoly

import import_data as impt
from class_DeepHit import Model_DeepHit
from get_main_CTD_1L import get_network_settings


OUTER = 2
BASE = Path("METABRIC/results_KAN1L_CTD")
FEATURE_FILE = Path("./sample data/METABRIC/cleaned_features_final.csv")
OUTDIR = Path("KAN1L_piecewise_key_edges_outer2")

# kind, feature_name, shared_idx, output_idx, label
EDGES = [
    ("CS", "age_at_diagnosis", None, 27, "age_to_c27"),
    ("CS", "NPI", None, 9, "NPI_to_c9"),
    ("SHARED", "age_at_diagnosis", None, 41, "age_to_s41"),
    ("SHARED", "NPI", None, 41, "NPI_to_s41"),
    ("CS_SHARED", None, 41, 19, "s41_to_c19"),
]


def load_normalizer(path):
    nf = np.load(path)
    return {
        "mode": str(nf["mode"]),
        "offset": nf["offset"],
        "scale": nf["scale"],
    }


def silu_np(x):
    return x / (1.0 + np.exp(-x))


def main():
    OUTDIR.mkdir(exist_ok=True)

    df = pd.read_csv(FEATURE_FILE)
    feature_names = list(df.columns)
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

    model = Model_DeepHit(
        input_dims,
        get_network_settings(hp, seed=1234 + OUTER),
    )
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    Xn = impt.apply_normalizer(raw_data, normalizer)
    Xdev = torch.tensor(Xn[dev_idx], dtype=torch.float32)

    with torch.no_grad():
        shared_out = model.shared_kan(Xdev)
        q = torch.cat([Xdev, shared_out], dim=1)

    shared = model.shared_kan
    cs = model.cs_kans[0]

    sh_layer = shared.act_fun[0]
    cs_layer = cs.act_fun[0]

    with torch.no_grad():
        _, _, sh_postacts, _ = sh_layer(Xdev)
        _, _, cs_postacts, _ = cs_layer(q)

    n_original = Xdev.shape[1]
    all_rows = []
    validation_rows = []

    print("\n" + "=" * 120)
    print("PIECEWISE-CUBIC EXPANSION OF KEY KAN EDGES — OUTER 2")
    print("=" * 120)
    print(
        "Each full edge is represented on each interval as:\n"
        "  phi(x) = mask * [scale_base*SiLU(v(x))"
        " + P3*(x-xL)^3 + P2*(x-xL)^2 + P1*(x-xL) + P0]\n"
        "where the polynomial coefficients already include scale_sp.\n"
        "For s41->c19, x means s41 itself and v(x)=x."
    )

    for kind, feature_name, shared_idx, out_idx, label in EDGES:
        if kind == "SHARED":
            in_idx = feature_names.index(feature_name)
            layer = sh_layer
            postacts = sh_postacts
            model_input = Xdev[:, in_idx].detach().cpu().numpy()
            raw_values = raw_data[dev_idx, in_idx]
            offset = float(normalizer["offset"][in_idx])
            scale = float(normalizer["scale"][in_idx])
            raw_x = raw_values
            v = model_input

        elif kind == "CS":
            in_idx = feature_names.index(feature_name)
            layer = cs_layer
            postacts = cs_postacts
            model_input = q[:, in_idx].detach().cpu().numpy()
            raw_values = raw_data[dev_idx, in_idx]
            offset = float(normalizer["offset"][in_idx])
            scale = float(normalizer["scale"][in_idx])
            raw_x = raw_values
            v = model_input

        elif kind == "CS_SHARED":
            in_idx = n_original + shared_idx
            layer = cs_layer
            postacts = cs_postacts
            model_input = q[:, in_idx].detach().cpu().numpy()
            raw_values = model_input.copy()
            offset = 0.0
            scale = 1.0
            raw_x = model_input.copy()
            v = model_input

        else:
            raise ValueError(kind)

        k = int(layer.k)
        if k != 3:
            raise RuntimeError(
                f"{label}: expected cubic k=3, got k={k}"
            )

        knots = layer.grid[in_idx, :].detach().cpu().numpy().astype(float)
        coef = layer.coef[in_idx, out_idx, :].detach().cpu().numpy().astype(float)
        mask = float(layer.mask[in_idx, out_idx].detach())
        scale_base = float(layer.scale_base[in_idx, out_idx].detach())
        scale_sp = float(layer.scale_sp[in_idx, out_idx].detach())

        # SciPy standard B-spline representation and piecewise polynomial.
        bs = BSpline(knots, coef, k, extrapolate=False)
        pp = PPoly.from_spline(bs)

        # Validate against the exact KAN edge outputs on development patients.
        spline_eval = bs(v)
        full_recon = mask * (
            scale_base * silu_np(v)
            + scale_sp * spline_eval
        )
        exact_edge = postacts[:, out_idx, in_idx].detach().cpu().numpy()

        valid = np.isfinite(full_recon)
        if not np.all(valid):
            raise RuntimeError(
                f"{label}: SciPy B-spline did not cover all observed inputs."
            )

        abs_err = np.abs(full_recon - exact_edge)

        validation_rows.append({
            "edge": label,
            "max_abs_error": float(np.max(abs_err)),
            "mean_abs_error": float(np.mean(abs_err)),
            "observed_x_min": float(np.min(raw_x)),
            "observed_x_max": float(np.max(raw_x)),
            "observed_v_min": float(np.min(v)),
            "observed_v_max": float(np.max(v)),
        })

        # PPoly uses local coordinates (v - knot_left).
        # Convert the spline term to local coordinates in raw x:
        #   v - vL = (x - xL)/scale
        # Then multiply by scale_sp.
        intervals_for_edge = []

        for m in range(len(pp.x) - 1):
            vL = float(pp.x[m])
            vR = float(pp.x[m + 1])

            if not (vR > vL + 1e-12):
                continue

            xL = offset + scale * vL
            xR = offset + scale * vR

            # Keep only intervals intersecting the observed range.
            obs_min = float(np.min(raw_x))
            obs_max = float(np.max(raw_x))
            if xR < obs_min - 1e-12 or xL > obs_max + 1e-12:
                continue

            # pp.c[:,m] is:
            # a3*(v-vL)^3 + a2*(v-vL)^2 + a1*(v-vL) + a0
            a3, a2, a1, a0 = [float(z) for z in pp.c[:, m]]

            P3 = mask * scale_sp * a3 / (scale ** 3)
            P2 = mask * scale_sp * a2 / (scale ** 2)
            P1 = mask * scale_sp * a1 / scale
            P0 = mask * scale_sp * a0

            row = {
                "edge": label,
                "kind": kind,
                "feature": feature_name if feature_name is not None else f"s{shared_idx}",
                "input_index": in_idx,
                "output_index": out_idx,
                "x_left": xL,
                "x_right": xR,
                "v_left": vL,
                "v_right": vR,
                "mask": mask,
                "scale_base": scale_base,
                "offset": offset,
                "scale": scale,
                "P3": P3,
                "P2": P2,
                "P1": P1,
                "P0": P0,
            }
            all_rows.append(row)
            intervals_for_edge.append(row)

        print("\n" + "-" * 120)
        print(label)
        print("-" * 120)
        print(
            f"Observed x range: [{np.min(raw_x):.6f}, {np.max(raw_x):.6f}]"
        )
        print(
            f"Validation vs KAN edge: max error={np.max(abs_err):.12e} | "
            f"mean error={np.mean(abs_err):.12e}"
        )
        print(
            f"Intervals covering observed range: {len(intervals_for_edge)}"
        )
        print(
            f"Base term: {mask:.10f} * {scale_base:.10f} * "
            f"SiLU((x-({offset:.10f}))/({scale:.10f}))"
        )

        preview = intervals_for_edge[:5]
        print("\nFIRST INTERVAL EQUATIONS")
        for r in preview:
            print(
                f"[{r['x_left']:.6f}, {r['x_right']:.6f}]: "
                f"phi(x) = {mask:.10f}*["
                f"{scale_base:.10f}*SiLU((x-({offset:.10f}))/({scale:.10f}))] "
                f"+ {r['P3']:+.10e}(x-({r['x_left']:.10f}))^3 "
                f"+ {r['P2']:+.10e}(x-({r['x_left']:.10f}))^2 "
                f"+ {r['P1']:+.10e}(x-({r['x_left']:.10f})) "
                f"+ {r['P0']:+.10e}"
            )

    intervals_df = pd.DataFrame(all_rows)
    validation_df = pd.DataFrame(validation_rows)

    intervals_df.to_csv(
        OUTDIR / "key_edges_piecewise_cubic_raw_scale.csv",
        index=False,
    )
    validation_df.to_csv(
        OUTDIR / "piecewise_validation.csv",
        index=False,
    )

    print("\n" + "=" * 120)
    print("VALIDATION SUMMARY")
    print(validation_df.to_string(index=False))
    print("\nFILES")
    print(OUTDIR / "key_edges_piecewise_cubic_raw_scale.csv")
    print(OUTDIR / "piecewise_validation.csv")
    print("=" * 120)


if __name__ == "__main__":
    main()
