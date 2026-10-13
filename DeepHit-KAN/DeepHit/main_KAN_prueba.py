# main_full_test.py
import os
import numpy as np
import torch

from utils import create_dataset
from MultKAN import KAN

# opcional: hypothesis
try:
    from hypothesis import detect_separability, get_molecule, plot_tree
    HAS_HYPOTHESIS = True
except Exception as e:
    HAS_HYPOTHESIS = False
    print("[WARN] hypothesis.py no importable:", repr(e))

torch.manual_seed(0)
np.random.seed(0)


def section(title: str):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def safe_call(name, fn, *args, **kwargs):
    try:
        out = fn(*args, **kwargs)
        print(f"[OK] {name}")
        return out
    except Exception as e:
        print(f"[FAIL] {name}: {repr(e)}")
        return None

def refine_and_fit(model, dataset, grid, steps=200, lamb=1e-3):
    model = model.refine(grid)
    model.fit(dataset, steps=steps, lamb=lamb)
    return model


def main():
    device = "cpu"

    # --------------------------------------------------------------------------------
    # A) DATASET SINTÉTICO CONTROLADO (verdad simbólica conocida)
    # --------------------------------------------------------------------------------
    section("A) Dataset sintético controlado (verdad conocida)")

    # f(x1, x2) = exp(sin(pi*x1) + x2^2)
    f = lambda x: torch.exp(torch.sin(torch.pi * x[:, [0]]) + x[:, [1]] ** 2)

    dataset = create_dataset(
        f,
        n_var=2,
        ranges=[-1, 1],
        train_num=2000,
        test_num=1000,
        normalize_input=False,
        normalize_label=False,
        seed=0,
        device=device,
    )

    print("train_input:", dataset["train_input"].shape)
    print("train_label:", dataset["train_label"].shape)
    print("test_input:", dataset["test_input"].shape)
    print("test_label:", dataset["test_label"].shape)

    # --------------------------------------------------------------------------------
    # B) KAN BASE: FIT / EVALUATE
    # --------------------------------------------------------------------------------
    section("B) KAN base: fit/evaluate")

    model = KAN(width=[2, 16, 1], grid=10, seed=0).to(device)

    _ = safe_call("model.fit(steps=200, lamb=1e-3)", model.fit, dataset, steps=200, lamb=1e-3)
    eval0 = safe_call("model.evaluate()", model.evaluate, dataset)
    if isinstance(eval0, dict):
        print("test_loss:", eval0.get("test_loss"))
        print("n_edge:", eval0.get("n_edge"), "n_grid:", eval0.get("n_grid"), "round/state:", getattr(model, "round", None), getattr(model, "state_id", None))

    # sanity forward
    x_test = dataset["test_input"][:8]
    y_pred = safe_call("forward(model(x))", model, x_test)
    if y_pred is not None:
        print("pred shape:", y_pred.shape, "| pred sample:", y_pred[:3].detach().cpu().numpy())

    # --------------------------------------------------------------------------------
    # C) REFINE GRID
    # --------------------------------------------------------------------------------
    section("C) refine(grid=20)")

    model = safe_call(
        "refine_and_fit(grid=20)",
        refine_and_fit,
        model,
        dataset,
        20
    ) or model

    eval1 = safe_call("model.evaluate() after refine", model.evaluate, dataset)

    # --------------------------------------------------------------------------------
    # D) PRUNE
    # --------------------------------------------------------------------------------
    section("D) prune(edge_th=1e-2, node_th=1e-2)")

    model = safe_call("model.prune()", model.prune, edge_th=1e-2, node_th=1e-2) or model
    eval2 = safe_call("model.evaluate() after prune", model.evaluate, dataset)
    if isinstance(eval2, dict):
        print("test_loss:", eval2.get("test_loss"))
        print("n_edge:", eval2.get("n_edge"), "n_grid:", eval2.get("n_grid"))

    # --------------------------------------------------------------------------------
    # E) REWIND (si existe)
    # --------------------------------------------------------------------------------
    section("E) rewind(state_id)")

    round_id = getattr(model, "round", None)
    state_id = getattr(model, "state_id", None)

    if round_id is None or state_id is None:
        print("[WARN] model.round o model.state_id no existen; salto rewind.")
    else:
        rewind_id = f"{round_id}.{state_id}"
        print(f"[INFO] rewinding to {rewind_id}")

        model_rew = safe_call(
            f"model.rewind('{rewind_id}')",
            model.rewind,
            rewind_id
        )

        if model_rew is not None:
            eval3 = safe_call(
                "model_rew.evaluate()",
                model_rew.evaluate,
                dataset
            )
            if isinstance(eval3, dict):
                print("rewind test_loss:", eval3.get("test_loss"))

    # --------------------------------------------------------------------------------
    # F) SYMBOLIC: fix_symbolic (si está implementado)
    # --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
# F) SYMBOLIC: fix_symbolic (API REAL de MultKAN)
# --------------------------------------------------------------------------------
    section("F) symbolic: fix_symbolic (API REAL de MultKAN)")

    if hasattr(model, "fix_symbolic"):
        # Solo fijamos lo que sabemos que corresponde a la función objetivo
        safe_call("fix_symbolic(0,0,0,'sin')", model.fix_symbolic, 0, 0, 0, "sin")
        safe_call("fix_symbolic(0,1,0,'x^2')", model.fix_symbolic, 0, 1, 0, "x^2")

        eval_sym = safe_call("model.evaluate() after fix_symbolic", model.evaluate, dataset)
        if isinstance(eval_sym, dict):
            print("test_loss:", eval_sym.get("test_loss"))
    else:
        print("[WARN] model.fix_symbolic no existe; salto.")


    # --------------------------------------------------------------------------------
    # G) AUTO SYMBOLIC (si existe)
    # --------------------------------------------------------------------------------
    section("G) auto_symbolic (si existe)")

    if hasattr(model, "auto_symbolic"):
        # Firma típica: model.auto_symbolic(lib=..., r2_th=..., ...)
        _ = safe_call("model.auto_symbolic()", model.auto_symbolic)
        eval_auto = safe_call("model.evaluate() after auto_symbolic", model.evaluate, dataset)
        if isinstance(eval_auto, dict):
            print("test_loss:", eval_auto.get("test_loss"))
    else:
        print("[WARN] model.auto_symbolic no existe en tu MultKAN; salto.")

    # --------------------------------------------------------------------------------
    # H) EXPRESIÓN / EXPORT SIMBÓLICO (si existe)
    # --------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------
# H) EXPRESIÓN / EXPORT SIMBÓLICO (FIX DEFINITIVO)
# --------------------------------------------------------------------------------
    section("H) exportar expresión simbólica")

    import sympy as sp
    import MultKAN as MK  # IMPORTA EL MÓDULO, no la clase

    input_dim = model.width_in[0]

    # 1) Definir en el namespace del MAIN (por si acaso)
    for i in range(1, input_dim + 1):
        globals()[f"x{i}"] = sp.Symbol(f"x{i}")

    # 2) Definir en el namespace de MultKAN (ESTE ES EL FIX REAL)
    for i in range(1, input_dim + 1):
        setattr(MK, f"x{i}", sp.Symbol(f"x{i}"))

    # 3) Ahora sí: export
    try:
        expr, _ = model.symbolic_formula()
        print("symbolic_formula() =>")
        print(expr)
    except Exception as e:
        print("[FAIL] model.symbolic_formula():", repr(e))


    # --------------------------------------------------------------------------------
    # I) HYPOTHESIS: separabilidad / árbol (si está)
    # --------------------------------------------------------------------------------
    section("I) hypothesis: separabilidad / árbol (si existe)")

    if not HAS_HYPOTHESIS:
        print("[WARN] hypothesis.py no está disponible/importable; salto.")
    else:
        # Usamos el modelo como función: f_model(x) -> y
        # algunos modelos devuelven (y, aux); si eso pasa, adaptamos
        def f_model(x):
            y = model(x)
            if isinstance(y, (tuple, list)):
                return y[0]
            return y

        x = dataset["test_input"][:200].detach()

        _ = safe_call("detect_separability(add)", detect_separability, f_model, x, mode="add", verbose=True)
        _ = safe_call("detect_separability(mul)", detect_separability, f_model, x, mode="mul", verbose=True)

        # plot_tree puede tardar; lo dejamos como opcional
        _ = safe_call("plot_tree()", plot_tree, f_model, x)

    # --------------------------------------------------------------------------------
    # J) PASO A DATOS MÉDICOS: SOLO PARA COMPROBAR QUE ENTRENAR NO ROMPE
    # (sin DeepHit todavía)
    # --------------------------------------------------------------------------------
    section("J) Smoke test con tu CSV médico: synthetic_comprisk.csv (solo KAN)")

    import pandas as pd

    csv_path = "synthetic_comprisk.csv"
    if not os.path.exists(csv_path):
        print(f"[WARN] No encuentro {csv_path} en el cwd. Colócalo junto a este script.")
        return

    df = pd.read_csv(csv_path)
    if not set(["time", "label"]).issubset(df.columns):
        print("[FAIL] El CSV no tiene columnas 'time' y 'event'. Columnas:", df.columns.tolist())
        return

    feature_cols = [c for c in df.columns if c not in ["time", "event"]]
    X = torch.tensor(df[feature_cols].values, dtype=torch.float32)
    y = torch.tensor(df["time"].values, dtype=torch.float32).unsqueeze(1)

    # normalización simple
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)

    # pequeño split
    n = X.shape[0]
    idx = torch.randperm(n)
    n_tr = int(0.8 * n)
    tr, te = idx[:n_tr], idx[n_tr:]

    dataset_med = {
        "train_input": X[tr].to(device),
        "train_label": y[tr].to(device),
        "test_input": X[te].to(device),
        "test_label": y[te].to(device),
    }

    model_med = KAN(width=[X.shape[1], 32, 1], grid=10, seed=0).to(device)

    # si tu KAN tiene .fit tipo repo, úsalo; si no, fallback manual
    if hasattr(model_med, "fit"):
        _ = safe_call("model_med.fit()", model_med.fit, dataset_med, steps=200, lamb=1e-3)
    else:
        print("[WARN] No hay model.fit() en KAN; usando loop manual MSE.")
        opt = torch.optim.Adam(model_med.parameters(), lr=1e-3)
        loss_fn = torch.nn.MSELoss()
        for epoch in range(200):
            pred = model_med(dataset_med["train_input"])
            if isinstance(pred, (tuple, list)):
                pred = pred[0]
            loss = loss_fn(pred, dataset_med["train_label"])
            loss.backward()
            opt.step()
            opt.zero_grad()
            if epoch % 50 == 0:
                print(f"epoch {epoch} | loss {loss.item():.4f}")

    eval_med = safe_call("model_med.evaluate()", model_med.evaluate, dataset_med) if hasattr(model_med, "evaluate") else None
    if isinstance(eval_med, dict):
        print("medical test_loss:", eval_med.get("test_loss"))

    # prune smoke
    if hasattr(model_med, "prune"):
        model_med = safe_call("model_med.prune()", model_med.prune, edge_th=1e-2, node_th=1e-2) or model_med
        _ = safe_call("model_med.forward after prune", model_med, dataset_med["test_input"][:8])

    section("DONE ✅  Si todo esto pasa, KAN está lista para integrarse en DeepHit.")


if __name__ == "__main__":
    main()
