"""XGBoost train vs validation log loss per boosting round (report 3.3 and question 9).

Only models that really produce an epoch-wise history get such a plot, so this covers XGBoost and
nothing else. "Validation" = the held-out trip-grouped CV fold of the TRAINING partition; the
student-test trips are not used here.

    python experiments/learning_curves.py --table features/candidates.parquet \
        [--params-json tuning/studies/xgboost_cv_macro_f1/best_params.json] [--out-dir experiments/results]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from person2 import config  # noqa: E402
from person2.experiments.harness import load_table  # noqa: E402
from person2.experiments.models import XGBOOST  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True, type=Path)
    ap.add_argument("--params-json", type=Path, default=None, help="best_params.json from tuning; default = untuned defaults")
    ap.add_argument("--label", default=None)
    ap.add_argument("--features", nargs="+", default=None, help="use only these feature columns")
    ap.add_argument("--seed", type=int, default=config.GLOBAL_SEED)
    ap.add_argument("--out-dir", type=Path, default=config.REPO_ROOT / "experiments" / "results")
    a = ap.parse_args()

    params = dict(XGBOOST.defaults)
    label = a.label or "xgboost (untuned defaults)"
    if a.params_json:
        params.update(json.loads(a.params_json.read_text())["params"])
        label = a.label or "xgboost (tuned)"

    df, _, cols = load_table(a.table, feature_cols=a.features)
    train = df[df.partition == "train"]
    X, y = train[cols].to_numpy(dtype="float32"), train.label.to_numpy()
    curves_tr, curves_va = [], []
    for k in sorted(train.fold.unique()):
        va = (train.fold == k).to_numpy()
        clf = XGBOOST.build(params, a.seed).named_steps["clf"]
        clf.fit(X[~va], y[~va], eval_set=[(X[~va], y[~va]), (X[va], y[va])], verbose=False)
        ev = clf.evals_result()
        curves_tr.append(ev["validation_0"]["logloss"])
        curves_va.append(ev["validation_1"]["logloss"])
        print(f"fold {k}: final train {curves_tr[-1][-1]:.4f}  val {curves_va[-1][-1]:.4f}  best val round {int(np.argmin(curves_va[-1])) + 1}")
    tr, va_ = np.array(curves_tr), np.array(curves_va)
    rounds = np.arange(1, tr.shape[1] + 1)

    a.out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for arr, name, c in ((tr, "train (the 4 training folds)", "C0"), (va_, "validation (the held-out fold)", "C1")):
        ax.plot(rounds, arr.mean(0), color=c, label=name)
        ax.fill_between(rounds, arr.min(0), arr.max(0), color=c, alpha=0.2)
    best = int(np.argmin(va_.mean(0))) + 1
    ax.axvline(best, color="grey", ls=":", label=f"min mean validation loss at round {best}")
    ax.set(xlabel="boosting round (shaded band = min and max over the 5 CV folds)", ylabel="log loss",
           title=f"{label}: train vs validation log loss")
    ax.legend()
    fig.tight_layout()
    stem = "xgboost_learning_curve" + ("_tuned" if a.params_json else "")
    fig.savefig(a.out_dir / f"{stem}.png", dpi=130)
    plt.close(fig)

    pd.DataFrame({"round": rounds, "train_logloss_mean": tr.mean(0), "val_logloss_mean": va_.mean(0)}).to_csv(
        a.out_dir / f"{stem}.csv", index=False)
    (a.out_dir / f"{stem}.json").write_text(json.dumps({
        "label": label, "params": params, "n_rounds": int(tr.shape[1]), "best_validation_round": best,
        "train_logloss_final": float(tr[:, -1].mean()), "val_logloss_final": float(va_[:, -1].mean()),
        "val_logloss_min": float(va_.mean(0).min()),
        "gap_final_val_minus_train": float(va_[:, -1].mean() - tr[:, -1].mean())}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
