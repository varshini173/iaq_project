"""
train_model_uci.py
-------------------
Trains and evaluates the air-quality classifier on the REAL UCI Air Quality
dataset (see prepare_uci_dataset.py for how the data is cleaned and labelled).
Uses scikit-learn only, so it runs on any laptop with no TensorFlow.

What it does
  1. Loads the data, splits it BY WEEK (train 70% / val 15% / test 15%) so
     neighbouring hours can't leak between train and test.
  2. Tunes the network (hidden layer sizes, L2 strength, class balancing) with
     5-fold week-grouped cross-validation on train+val. The test set is not touched.
  3. Trains the final model with early stopping on the validation weeks.
  4. Evaluates ONCE on the held-out test weeks; compares against simple
     baselines; runs a chronological-split stress test.
  5. Exports everything the rest of the project needs to models/uci/:
       model_uci.json, model_weights_uci.h, metrics.json, report.md,
       confusion_matrix.png, training_curve.png, test_replay.csv

Usage:
    python train_model_uci.py            # full search (a few minutes)
    python train_model_uci.py --quick    # tiny search, for a fast smoke test
"""

import argparse
import copy
import itertools
import json
import os
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score, log_loss,
                             precision_recall_fscore_support)
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPClassifier

import prepare_uci_dataset as P

warnings.filterwarnings("ignore")

OUT_DIR = os.path.join(P.HERE, "models", "uci")
SEED = 42
MAX_EPOCHS = 300
PATIENCE = 20
BATCH = 64
LR = 1e-3
N_CLASSES = len(P.LABEL_NAMES)


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def oversample(X, y, rng):
    """Randomly duplicate minority-class rows until every class has as many rows as the biggest."""
    counts = np.bincount(y, minlength=N_CLASSES)
    target = counts.max()
    idx = []
    for c in range(N_CLASSES):
        ci = np.where(y == c)[0]
        idx.append(ci)
        if len(ci) < target:
            idx.append(rng.choice(ci, size=target - len(ci), replace=True))
    idx = np.concatenate(idx)
    rng.shuffle(idx)
    return X[idx], y[idx]


def fit_mlp(Xtr, ytr, Xva, yva, hidden, alpha, balance, seed):
    """
    Train an MLP one epoch at a time (partial_fit) and keep the weights from the
    epoch with the lowest validation log-loss. sklearn's built-in early_stopping
    picks its validation rows at random, which leaks neighbouring hours - so
    we do our own early stopping on properly separated validation weeks.
    Returns (best_model, history dict).
    """
    rng = np.random.default_rng(seed)
    if balance:
        Xtr, ytr = oversample(Xtr, ytr, rng)
    clf = MLPClassifier(hidden_layer_sizes=hidden, activation="relu", solver="adam", alpha=alpha,
                        batch_size=BATCH, learning_rate_init=LR, random_state=seed)
    classes = np.arange(N_CLASSES)
    best, best_loss, best_epoch, bad = None, np.inf, 0, 0
    hist = {"train_loss": [], "val_loss": [], "val_acc": []}
    for epoch in range(1, MAX_EPOCHS + 1):
        perm = rng.permutation(len(Xtr))
        clf.partial_fit(Xtr[perm], ytr[perm], classes=classes)
        p = clf.predict_proba(Xva)
        vl = log_loss(yva, p, labels=classes)
        hist["train_loss"].append(float(clf.loss_))
        hist["val_loss"].append(float(vl))
        hist["val_acc"].append(float(accuracy_score(yva, p.argmax(1))))
        if vl < best_loss - 1e-5:
            best_loss, best_epoch, bad = vl, epoch, 0
            best = copy.deepcopy(clf)
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    hist["best_epoch"] = best_epoch
    return best, hist


def metrics(y, pred):
    p, r, f, _ = precision_recall_fscore_support(y, pred, average="macro", zero_division=0)
    return {"accuracy": float(accuracy_score(y, pred)), "precision_macro": float(p),
            "recall_macro": float(r), "f1_macro": float(f)}


def predict_proba_numpy(coefs, intercepts, Xn):
    """Same forward pass the dashboard/firmware runs (ReLU hidden layers, softmax out)."""
    h = Xn
    for i, (W, b) in enumerate(zip(coefs, intercepts)):
        h = h @ W + b
        if i < len(coefs) - 1:
            h = np.maximum(h, 0)
    h = h - h.max(axis=1, keepdims=True)
    e = np.exp(h)
    return e / e.sum(axis=1, keepdims=True)


# ----------------------------------------------------------------------------
# hyper-parameter search
# ----------------------------------------------------------------------------
def search(X, y, groups, grid):
    gkf = GroupKFold(n_splits=5)
    rows = []
    for cfg in grid:
        t0 = time.time()
        f1s, accs, epochs = [], [], []
        for k, (tr, va) in enumerate(gkf.split(X, y, groups)):
            mu, sd = X[tr].mean(0), X[tr].std(0)
            sd[sd == 0] = 1
            Xtr, Xva = (X[tr] - mu) / sd, (X[va] - mu) / sd
            model, hist = fit_mlp(Xtr, y[tr], Xva, y[va], cfg["hidden"], cfg["alpha"], cfg["balance"], SEED + k)
            pred = model.predict(Xva)
            f1s.append(f1_score(y[va], pred, average="macro"))
            accs.append(accuracy_score(y[va], pred))
            epochs.append(hist["best_epoch"])
        row = {**cfg, "cv_f1_macro": float(np.mean(f1s)), "cv_f1_std": float(np.std(f1s)),
               "cv_accuracy": float(np.mean(accs)), "mean_best_epoch": float(np.mean(epochs))}
        rows.append(row)
        print(f"  hidden={str(cfg['hidden']):12s} alpha={cfg['alpha']:<7g} balance={str(cfg['balance']):5s} "
              f"-> CV macro-F1 {row['cv_f1_macro']:.3f} (+/-{row['cv_f1_std']:.3f})  "
              f"acc {row['cv_accuracy']:.3f}   [{time.time() - t0:.0f}s]", flush=True)
    return rows


# ----------------------------------------------------------------------------
# exports
# ----------------------------------------------------------------------------
def export_c_header(coefs, intercepts, mean, std, path):
    sizes = [coefs[0].shape[0]] + [c.shape[1] for c in coefs]
    L = ["// AUTO-GENERATED by train_model_uci.py - do not hand-edit.",
         "// MLP (ReLU hidden layers, softmax output) trained on the UCI Air Quality dataset.",
         "// Inputs (in order): " + ", ".join(P.FEATURES),
         "// NOTE: these inputs are the UCI dataset's sensor array, NOT a DHT22 + MQ-135.",
         "#pragma once\n",
         f"#define IAQ_N_FEATURES {sizes[0]}",
         f"#define IAQ_N_LAYERS {len(coefs)}",
         f"#define IAQ_N_CLASSES {sizes[-1]}\n",
         "static const int iaq_layer_sizes[IAQ_N_LAYERS + 1] = {" + ", ".join(map(str, sizes)) + "};\n",
         "static const float iaq_feature_mean[IAQ_N_FEATURES] = {" + ", ".join(f"{m:.6f}f" for m in mean) + "};",
         "static const float iaq_feature_std[IAQ_N_FEATURES]  = {" + ", ".join(f"{s:.6f}f" for s in std) + "};\n"]
    for i, (W, b) in enumerate(zip(coefs, intercepts)):
        L.append(f"// Layer {i}: {W.shape[0]} -> {W.shape[1]}")
        L.append(f"static const float iaq_W{i}[{W.shape[0]}][{W.shape[1]}] = {{")
        for row in W:
            L.append("  {" + ", ".join(f"{v:.6f}f" for v in row) + "},")
        L.append("};")
        L.append(f"static const float iaq_b{i}[{W.shape[1]}] = {{" + ", ".join(f"{v:.6f}f" for v in b) + "};\n")
    L.append('static const char* iaq_class_names[IAQ_N_CLASSES] = {' + ", ".join(f'"{n}"' for n in P.LABEL_NAMES) + "};")
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")


def make_plots(cm, hist, path_cm, path_curve):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.6, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(N_CLASSES), P.LABEL_NAMES)
    ax.set_yticks(range(N_CLASSES), P.LABEL_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual (reference analyzer)")
    ax.set_title("Test-set confusion matrix")
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(path_cm, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    ax.plot(hist["train_loss"], label="train loss")
    ax.plot(hist["val_loss"], label="validation loss")
    ax.axvline(hist["best_epoch"] - 1, ls="--", c="grey", lw=1, label=f"best epoch ({hist['best_epoch']})")
    ax.set_xlabel("epoch")
    ax.set_ylabel("log-loss")
    ax.legend()
    ax.set_title("Training curve")
    fig.tight_layout()
    fig.savefig(path_curve, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="tiny search for a smoke test")
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    df = P.load_clean()
    X = df[P.FEATURES].to_numpy(dtype=np.float64)
    y = df["label"].to_numpy(dtype=np.int64)
    tr, va, te = P.week_block_split(df)
    print(f"Rows: {len(df)}   train {tr.sum()}  val {va.sum()}  test {te.sum()}")
    print("Class share (train): " + ", ".join(f"{n} {np.mean(y[tr] == i):.1%}" for i, n in enumerate(P.LABEL_NAMES)))

    # ---- 1. hyper-parameter search on train+val weeks only -----------------
    if args.quick:
        grid_axes = dict(hidden=[(16, 8)], alpha=[1e-3], balance=[False])
    else:
        grid_axes = dict(hidden=[(8,), (16, 8), (32, 16), (32, 16, 8)], alpha=[1e-4, 1e-2], balance=[False, True])
    grid = [dict(zip(grid_axes, v)) for v in itertools.product(*grid_axes.values())]
    print(f"\nCross-validated search over {len(grid)} configurations (5 week-grouped folds each)...")
    trva = tr | va
    rows = search(X[trva], y[trva], df["week"].to_numpy()[trva], grid)
    best_cfg = max(rows, key=lambda r: (round(r["cv_f1_macro"], 3), r["cv_accuracy"]))
    cfg = {k: best_cfg[k] for k in ("hidden", "alpha", "balance")}
    print(f"\nBest configuration: {cfg}")

    # ---- 2. final training: normalise on train, early-stop on val ----------
    mean, std = X[tr].mean(0), X[tr].std(0)
    std[std == 0] = 1
    Xn = (X - mean) / std
    best_model, best_hist, best_vl = None, None, np.inf
    for s in range(5):  # 5 random initialisations; keep the best on VALIDATION (never on test)
        m, h = fit_mlp(Xn[tr], y[tr], Xn[va], y[va], cfg["hidden"], cfg["alpha"], cfg["balance"], SEED + 100 + s)
        vl = min(h["val_loss"])
        print(f"  init {s}: best val log-loss {vl:.4f} at epoch {h['best_epoch']}")
        if vl < best_vl:
            best_model, best_hist, best_vl = m, h, vl
    coefs, intercepts = best_model.coefs_, best_model.intercepts_
    n_params = sum(c.size for c in coefs) + sum(b.size for b in intercepts)

    # ---- 3. evaluation on held-out test weeks -------------------------------
    result = {"config": cfg, "n_parameters": int(n_params), "best_epoch": best_hist["best_epoch"]}
    for name, mask in (("train", tr), ("validation", va), ("test", te)):
        result[name] = metrics(y[mask], best_model.predict(Xn[mask]))
    pred_te = best_model.predict(Xn[te])
    cm = confusion_matrix(y[te], pred_te, labels=range(N_CLASSES))
    result["test"]["confusion_matrix"] = cm.tolist()
    rep = classification_report(y[te], pred_te, target_names=P.LABEL_NAMES, zero_division=0, output_dict=True)
    result["test"]["per_class"] = {n: {k: float(rep[n][k]) for k in ("precision", "recall", "f1-score", "support")}
                                   for n in P.LABEL_NAMES}

    # sanity: the numpy forward pass (what dashboard + firmware do) must match sklearn
    ours = predict_proba_numpy(coefs, intercepts, Xn[te])
    assert np.allclose(ours, best_model.predict_proba(Xn[te]), atol=1e-6), "exported maths != sklearn"
    result["export_check_max_abs_diff"] = float(np.abs(ours - best_model.predict_proba(Xn[te])).max())

    # ---- 4. baselines (same split) ------------------------------------------
    base = {}
    for name, clf in (("Majority class", DummyClassifier(strategy="most_frequent")),
                      ("Logistic regression", LogisticRegression(max_iter=3000)),
                      ("Gradient boosting (reference, too big for MCU)", HistGradientBoostingClassifier(random_state=SEED))):
        clf.fit(Xn[tr], y[tr])
        base[name] = metrics(y[te], clf.predict(Xn[te]))
    result["baselines_test"] = base

    # ---- 5. stress test: chronological split --------------------------------
    order = np.argsort(df["timestamp"].to_numpy())
    n = len(order)
    c_tr, c_va, c_te = order[:int(.70 * n)], order[int(.70 * n):int(.85 * n)], order[int(.85 * n):]
    cm_mu, cm_sd = X[c_tr].mean(0), X[c_tr].std(0)
    cm_sd[cm_sd == 0] = 1
    Xc = (X - cm_mu) / cm_sd
    m_c, _ = fit_mlp(Xc[c_tr], y[c_tr], Xc[c_va], y[c_va], cfg["hidden"], cfg["alpha"], cfg["balance"], SEED)
    result["chronological_stress_test"] = {
        "note": "train on the earliest 70% of hours, test on the latest 15% (Feb-Apr 2005). Sensors drift and "
                "seasons change, so this is harder than the week-block split.",
        "test_period": [str(df['timestamp'].iloc[c_te].min()), str(df['timestamp'].iloc[c_te].max())],
        **metrics(y[c_te], m_c.predict(Xc[c_te]))}

    result["search"] = rows
    result["data"] = {"rows": int(len(df)), "train": int(tr.sum()), "val": int(va.sum()), "test": int(te.sum()),
                      "class_share": {n: float(np.mean(y == i)) for i, n in enumerate(P.LABEL_NAMES)}}

    # ---- 6. write artefacts --------------------------------------------------
    model_json = {
        "features": P.FEATURES, "feature_labels": P.FEATURE_LABELS, "classes": P.LABEL_NAMES,
        "mean": mean.tolist(), "std": std.tolist(),
        "layers": [{"W": W.tolist(), "b": b.tolist()} for W, b in zip(coefs, intercepts)],
        "test_metrics": result["test"], "config": {"hidden": list(cfg["hidden"]), "alpha": cfg["alpha"],
                                                      "balance": cfg["balance"]},
    }
    with open(os.path.join(OUT_DIR, "model_uci.json"), "w") as f:
        json.dump(model_json, f)
    export_c_header(coefs, intercepts, mean, std, os.path.join(OUT_DIR, "model_weights_uci.h"))
    with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
        json.dump(result, f, indent=2, default=lambda o: list(o) if isinstance(o, tuple) else str(o))
    make_plots(cm, best_hist, os.path.join(OUT_DIR, "confusion_matrix.png"), os.path.join(OUT_DIR, "training_curve.png"))

    # held-out rows the dashboard replays (never seen in training or tuning)
    replay = df.loc[te, ["timestamp"] + P.FEATURES + ["CO(GT)", "NO2(GT)", "label"]]
    replay.to_csv(os.path.join(OUT_DIR, "test_replay.csv"), index=False)

    write_report(result, os.path.join(OUT_DIR, "report.md"))

    t = result["test"]
    print("\n================ FINAL (held-out test weeks) ================")
    print(f"Accuracy {t['accuracy']:.4f} | macro precision {t['precision_macro']:.4f} | "
          f"recall {t['recall_macro']:.4f} | F1 {t['f1_macro']:.4f}")
    print("Confusion matrix (rows=actual, cols=predicted; Good/Moderate/Poor):")
    print(cm)
    print(classification_report(y[te], pred_te, target_names=P.LABEL_NAMES, zero_division=0))
    for k, v in base.items():
        print(f"baseline {k:50s} acc {v['accuracy']:.3f}  macro-F1 {v['f1_macro']:.3f}")
    cs = result["chronological_stress_test"]
    print(f"chronological stress test (last 15% of time): acc {cs['accuracy']:.3f}  macro-F1 {cs['f1_macro']:.3f}")
    print(f"\nModel: {cfg['hidden']}  {n_params} parameters. Artefacts written to {OUT_DIR}")


def write_report(r, path):
    t, d = r["test"], r["data"]
    L = ["# UCI Air Quality model - results", "",
         f"Rows after cleaning: **{d['rows']}** (train {d['train']} / val {d['val']} / test {d['test']}, split by calendar week).",
         "Class share: " + ", ".join(f"{k} {v:.1%}" for k, v in d["class_share"].items()) + ".", "",
         f"Chosen network: hidden layers {tuple(r['config']['hidden'])}, L2 alpha {r['config']['alpha']}, "
         f"class balancing {'on' if r['config']['balance'] else 'off'}; **{r['n_parameters']} parameters**; "
         f"best epoch {r['best_epoch']}.", "",
         "## Held-out test weeks", "",
         "| Metric | Value |", "|---|---|",
         f"| Accuracy | {t['accuracy']:.4f} |", f"| Precision (macro) | {t['precision_macro']:.4f} |",
         f"| Recall (macro) | {t['recall_macro']:.4f} |", f"| F1 (macro) | {t['f1_macro']:.4f} |", "",
         "Confusion matrix (rows = actual, columns = predicted; Good / Moderate / Poor):", "", "```"]
    L += [str(row) for row in t["confusion_matrix"]]
    L += ["```", "", "| Class | Precision | Recall | F1 | Support |", "|---|---|---|---|---|"]
    for n, v in t["per_class"].items():
        L.append(f"| {n} | {v['precision']:.3f} | {v['recall']:.3f} | {v['f1-score']:.3f} | {int(v['support'])} |")
    L += ["", "## Baselines on the same test weeks", "", "| Model | Accuracy | Macro-F1 |", "|---|---|---|"]
    for k, v in r["baselines_test"].items():
        L.append(f"| {k} | {v['accuracy']:.3f} | {v['f1_macro']:.3f} |")
    L.append(f"| **This MLP** | **{t['accuracy']:.3f}** | **{t['f1_macro']:.3f}** |")
    cs = r["chronological_stress_test"]
    L += ["", "## Stress test: train on the past, test on the future", "",
          f"{cs['note']}", "", f"Test period {cs['test_period'][0]} to {cs['test_period'][1]}: "
          f"accuracy **{cs['accuracy']:.3f}**, macro-F1 **{cs['f1_macro']:.3f}**.", "",
          "## Hyper-parameter search (5-fold, week-grouped CV on train+val)", "",
          "| Hidden | alpha | Balanced | CV macro-F1 | CV accuracy |", "|---|---|---|---|---|"]
    for row in sorted(r["search"], key=lambda x: -x["cv_f1_macro"]):
        L.append(f"| {tuple(row['hidden'])} | {row['alpha']:g} | {row['balance']} | {row['cv_f1_macro']:.3f} | {row['cv_accuracy']:.3f} |")
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
