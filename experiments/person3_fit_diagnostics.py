"""
Overfitting check for Person 3's models (report 3.2, question C9): the same
model scored on the trips it was trained on (in-sample) versus on held-out trips
(trip-grouped CV, from the harness run's own metrics.json). A large in-sample
advantage means the model memorises its training trips.

    python experiments/person3_fit_diagnostics.py --table <team table> \
        --run person2/experiments/logs/<rf run> --run person2/experiments/logs/<svm run> ...

Writes experiments/results/person3_fit_diagnostics.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd  # noqa: E402

from person2.experiments.harness import _fit, load_table  # noqa: E402
from person2.experiments.metrics import candidate_metrics  # noqa: E402
from person2.experiments.models import REGISTRY  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", type=Path, required=True)
    ap.add_argument("--run", type=Path, action="append", required=True, help="harness run folder (repeatable)")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "experiments" / "results" / "person3_fit_diagnostics.json")
    a = ap.parse_args()

    df, _, _ = load_table(a.table)
    train = df[df.partition == "train"]
    out = {}
    for run in a.run:
        cfg = json.loads((run / "config.json").read_text())
        met = json.loads((run / "metrics.json").read_text())
        cols = cfg["feature_cols"]
        X, y = train[cols].to_numpy(dtype="float32"), train["label"].to_numpy()
        model = _fit(REGISTRY[cfg["model"]].build(cfg["params"], cfg["seed"]), X, y,
                     train.trip_id.to_numpy(), cfg["calibration"])
        ins = candidate_metrics(y, model.predict_proba(X)[:, 1], cfg["threshold"])
        cv = met["cv_pooled"]
        out[cfg["run_id"]] = {
            "model": cfg["model"], "calibration": cfg["calibration"], "params": cfg["params"],
            "in_sample": {k: ins[k] for k in ("macro_f1", "log_loss", "pos_recall", "pos_precision")},
            "cv_out_of_fold": {k: cv[k] for k in ("macro_f1", "log_loss", "pos_recall", "pos_precision")},
            "student_test": {k: met["student_test"][k] for k in ("macro_f1", "log_loss", "top1_acc")},
            "gap_macro_f1_in_sample_minus_cv": ins["macro_f1"] - cv["macro_f1"],
        }
        print(cfg["run_id"], json.dumps(out[cfg["run_id"]]["in_sample"]), "vs CV", json.dumps(out[cfg["run_id"]]["cv_out_of_fold"]))
    a.out.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
