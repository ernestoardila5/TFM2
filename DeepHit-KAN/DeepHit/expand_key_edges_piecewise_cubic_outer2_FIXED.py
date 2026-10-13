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
    x = np.asarray(x, dtype=np.float64)
    return x / (1.0 + np.exp(-x))


def poly_add(a, b):
    n = max(len(a), len(b))
    out = np.zeros(n, dtype=np.float64)
    out[:len(a)] += a
    out[:len(b)] += b
    return out


def poly_mul_linear(p, c0, c1):
    # p(y) * (c0 + c1*y), coefficients in ascending powers.
    out = np.zeros(len(p) + 1, dtype=np.float64)
    out[:len(p)] += c0 * p
    out[1:] += c1 * p
    return out


def spline_piece_local(knots, coef, degree, interval_idx):
    """
    Exact Cox-de Boor construction on ONE knot interval.

    Returns coefficients [a0,a1,...,ak] such that, on
        knots[m] <= v < knots[m+1],
    the KAN spline equals
        a0 + a1*y + ... + ak*y^k
    where y = v - knots[m].

    This follows the same recursion as spline.py::B_batch rather than
    SciPy's restricted "base interval" convention.
    """
    t = np.asarray(knots, dtype=np.float64)
    c = np.asarray(coef, dtype=np.float64)
    m = int(interval_idx)
    vL = float(t[m])

    # Degree-0 basis functions on this particular interval.
    polys = []
    for i in range(len(t) - 1):
        if i == m and t[i + 1] > t[i]:
            polys.append(np.array([1.0], dtype=np.float64))
        else:
            polys.append(np.array([0.0], dtype=np.float64))

    # Cox-de Boor recursion.
    for d in range(1, degree + 1):
        new_polys = []
        for i in range(len(t) - d - 1):
            out = np.zeros(d + 1, dtype=np.float64)

            den1 = t[i + d] - t[i]
            if abs(den1) > 1e-15:
                # (v - t_i)/den1
                # v = vL + y
                term1 = poly_mul_linear(
                    polys[i],
                    (vL - t[i]) / den1,
                    1.0 / den1,
                )
                out[:len(term1)] += term1

            den2 = t[i + d + 1] - t[i + 1]
            if abs(den2) > 1e-15:
                # (t_{i+d+1} - v)/den2
                # v = vL + y
                term2 = poly_mul_linear(
                    polys[i + 1],
                    (t[i + d + 1] - vL) / den2,
                    -1.0 / den2,
                )
                out[:len(term2)] += term2

            new_polys.append(out)

        polys = new_polys

    if len(polys) != len(c):
        raise RuntimeError(
            f"Basis count {len(polys)} != coefficient count {len(c)}"
        )

    spline_poly = np.zeros(degree + 1, dtype=np.float64)
    for ci, pi in zip(c, polys):
        spline_poly[:len(pi)] += ci * pi

    return spline_poly


def build_all_piece_polys(knots, coef, degree):
    pieces = {}
    for m in range(len(knots) - 1):
        if knots[m + 1] <= knots[m] + 1e-15:
            continue
        pieces[m] = spline_piece_local(
            knots, coef, degree, m
        )
    return pieces


def eval_piecewise_spline(v_values, knots, pieces):
    vals = np.zeros(len(v_values), dtype=np.float64)

    for n, v in enumerate(v_values):
        # B_batch uses [left, right); outside full extended grid => zero.
        m = np.searchsorted(knots, v, side="right") - 1

        if m < 0 or m >= len(knots) - 1:
            vals[n] = 0.0
            continue

        if m not in pieces:
            vals[n] = 0.0
            continue

        y = float(v - knots[m])
        a = pieces[m]  # ascending [a0,a1,a2,a3]
        vals[n] = sum(a[p] * (y ** p) for p in range(len(a)))

    return vals


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
    model.load_state_dict(
        torch.load(model_path, map_location="cpu")
    )
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
        "This version follows the SAME Cox-de Boor recursion as spline.py/B_batch,\n"
        "including the KAN's EXTENDED grid outside the core knots.\n"
    )
    print(
        "On each interval:\n"
        "  phi(x) = scale_base*SiLU(v(x))\n"
        "           + P3*(x-xL)^3 + P2*(x-xL)^2\n"
        "           + P1*(x-xL) + P0\n"
        "The P coefficients already include mask*scale_sp."
    )

    for kind, feature_name, shared_idx, out_idx, label in EDGES:
        if kind == "SHARED":
            in_idx = feature_names.index(feature_name)
            layer = sh_layer
            postacts = sh_postacts
            v = Xdev[:, in_idx].detach().cpu().numpy().astype(np.float64)
            raw_x = raw_data[dev_idx, in_idx].astype(np.float64)
            offset = float(normalizer["offset"][in_idx])
            scale = float(normalizer["scale"][in_idx])

        elif kind == "CS":
            in_idx = feature_names.index(feature_name)
            layer = cs_layer
            postacts = cs_postacts
            v = q[:, in_idx].detach().cpu().numpy().astype(np.float64)
            raw_x = raw_data[dev_idx, in_idx].astype(np.float64)
            offset = float(normalizer["offset"][in_idx])
            scale = float(normalizer["scale"][in_idx])

        elif kind == "CS_SHARED":
            in_idx = n_original + shared_idx
            layer = cs_layer
            postacts = cs_postacts
            v = q[:, in_idx].detach().cpu().numpy().astype(np.float64)
            raw_x = v.copy()       # here x is s41 itself
            offset = 0.0
            scale = 1.0

        else:
            raise ValueError(kind)

        degree = int(layer.k)
        if degree != 3:
            raise RuntimeError(
                f"{label}: expected k=3, got k={degree}"
            )

        knots = (
            layer.grid[in_idx, :]
            .detach().cpu().numpy().astype(np.float64)
        )
        coef = (
            layer.coef[in_idx, out_idx, :]
            .detach().cpu().numpy().astype(np.float64)
        )

        mask = float(layer.mask[in_idx, out_idx].detach())
        scale_base = float(
            layer.scale_base[in_idx, out_idx].detach()
        )
        scale_sp = float(
            layer.scale_sp[in_idx, out_idx].detach()
        )

        pieces = build_all_piece_polys(
            knots, coef, degree
        )

        # Exact KAN-style spline reconstruction on observed points.
        spline_recon = eval_piecewise_spline(
            v, knots, pieces
        )

        full_recon = mask * (
            scale_base * silu_np(v)
            + scale_sp * spline_recon
        )

        exact_edge = (
            postacts[:, out_idx, in_idx]
            .detach().cpu().numpy().astype(np.float64)
        )

        abs_err = np.abs(full_recon - exact_edge)

        full_grid_raw = offset + scale * knots

        validation_rows.append({
            "edge": label,
            "max_abs_error": float(np.max(abs_err)),
            "mean_abs_error": float(np.mean(abs_err)),
            "observed_x_min": float(np.min(raw_x)),
            "observed_x_max": float(np.max(raw_x)),
            "full_extended_grid_x_min": float(np.min(full_grid_raw)),
            "full_extended_grid_x_max": float(np.max(full_grid_raw)),
            "core_grid_x_min": float(
                offset + scale * knots[degree]
            ),
            "core_grid_x_max": float(
                offset + scale * knots[-degree - 1]
            ),
        })

        # Convert every relevant v-interval polynomial to raw-x scale.
        intervals_for_edge = []
        obs_min = float(np.min(raw_x))
        obs_max = float(np.max(raw_x))

        for m, a in pieces.items():
            vL = float(knots[m])
            vR = float(knots[m + 1])

            xL = offset + scale * vL
            xR = offset + scale * vR

            if xR < obs_min - 1e-12 or xL > obs_max + 1e-12:
                continue

            # spline(v) = a0 + a1*(v-vL)+a2*(v-vL)^2+a3*(v-vL)^3
            # v-vL = (x-xL)/scale
            P0 = mask * scale_sp * a[0]
            P1 = mask * scale_sp * a[1] / scale
            P2 = mask * scale_sp * a[2] / (scale ** 2)
            P3 = mask * scale_sp * a[3] / (scale ** 3)

            row = {
                "edge": label,
                "kind": kind,
                "feature": (
                    feature_name
                    if feature_name is not None
                    else f"s{shared_idx}"
                ),
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
            f"Observed x range      : "
            f"[{np.min(raw_x):.6f}, {np.max(raw_x):.6f}]"
        )
        print(
            f"Core-knot x range     : "
            f"[{offset + scale*knots[degree]:.6f}, "
            f"{offset + scale*knots[-degree-1]:.6f}]"
        )
        print(
            f"FULL extended x range : "
            f"[{np.min(full_grid_raw):.6f}, "
            f"{np.max(full_grid_raw):.6f}]"
        )
        print(
            "Validation vs actual KAN edge: "
            f"max error={np.max(abs_err):.12e} | "
            f"mean error={np.mean(abs_err):.12e}"
        )
        print(
            f"Nonzero knot intervals intersecting observed range: "
            f"{len(intervals_for_edge)}"
        )

        print("\nFIRST 5 INTERVAL EQUATIONS")
        for r in intervals_for_edge[:5]:
            print(
                f"[{r['x_left']:.6f}, {r['x_right']:.6f}):\n"
                f"  phi(x) = {mask*scale_base:+.10f}"
                f"*SiLU((x-({offset:.10f}))/({scale:.10f}))\n"
                f"           {r['P3']:+.10e}*(x-({r['x_left']:.10f}))^3\n"
                f"           {r['P2']:+.10e}*(x-({r['x_left']:.10f}))^2\n"
                f"           {r['P1']:+.10e}*(x-({r['x_left']:.10f}))\n"
                f"           {r['P0']:+.10e}"
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
    print(
        OUTDIR / "key_edges_piecewise_cubic_raw_scale.csv"
    )
    print(
        OUTDIR / "piecewise_validation.csv"
    )
    print("=" * 120)


if __name__ == "__main__":
    main()
