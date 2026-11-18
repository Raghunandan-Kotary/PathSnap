"""Consistent timing of every model (report 3.1 'Inference time' column and section 4).

Timings taken inside separate training runs are noisy (they depend on what else the machine is doing and on
one-off start-up costs), so this script times all models together, one after another, on the same table:
each model is fitted once on the full training set (fit time recorded), then `predict_proba` is run on the
complete student-test candidate set 5 times after a warm-up call; the median is reported.

    python experiments/time_models.py --table <candidates_features.parquet> --features <5 columns> \
        [--tuned-params tuning/studies/xgboost_cv_log_loss/best_params.json]

Run it while nothing else is using the CPU. Output: experiments/results/timing.csv and timing.json.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from person2 import config  # noqa: E402
from person2.experiments.harness import load_table  # noqa: E402
from person2.experiments.models import GAUSSIAN_NB, HGB, LINEAR_SVM, LOGREG, RANDOM_FOREST, XGBOOST  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True, type=Path)
    ap.add_argument("--features", nargs="+", default=None)
    ap.add_argument("--tuned-params", type=Path, default=None)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--only", nargs="+", default=None, help="time only these labels and update their rows in an existing timing.csv")
    ap.add_argument("--seed", type=int, default=config.GLOBAL_SEED)
    ap.add_argument("--out-dir", type=Path, default=config.REPO_ROOT / "experiments" / "results")
    a = ap.parse_args()

    df, _, cols = load_table(a.table, feature_cols=a.features)
    tr, te = df[df.partition == "train"], df[df.partition == "student_test"]
    X, y = tr[cols].to_numpy("float32"), tr.label.to_numpy()
    Xt = te[cols].to_numpy("float32")
    n_trips, n_cand = te.trip_id.nunique(), len(te)

    plan = [(s.name, s, {}) for s in (XGBOOST, GAUSSIAN_NB, LOGREG, RANDOM_FOREST, HGB, LINEAR_SVM)]
    if a.tuned_params:
        plan.insert(1, ("xgboost_tuned", XGBOOST, json.loads(a.tuned_params.read_text())["params"]))

    if a.only:
        plan = [x for x in plan if x[0] in a.only]
    rows = []
    for label, spec, params in plan:
        model = spec.build({**spec.defaults, **params}, a.seed)
        t0 = time.perf_counter()
        model.fit(X, y)
        fit_s = time.perf_counter() - t0
        model.predict_proba(Xt[:256])  # warm-up
        reps = []
        for _ in range(a.repeats):
            t0 = time.perf_counter()
            model.predict_proba(Xt)
            reps.append(time.perf_counter() - t0)
        med = float(np.median(reps))
        rows.append({"label": label, "fit_time_s": fit_s, "predict_median_s": med, "predict_min_s": min(reps), "predict_max_s": max(reps),
                     "infer_ms_per_trip": 1000 * med / n_trips, "infer_ms_per_1k_candidates": 1e6 * med / n_cand})
        print(f"{label:24s} fit {fit_s:7.1f}s | predict median {med * 1000:8.1f} ms (min {min(reps) * 1000:.0f}, max {max(reps) * 1000:.0f}) "
              f"| {1000 * med / n_trips:6.1f} ms/trip", flush=True)

    out = pd.DataFrame(rows)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    if a.only and (a.out_dir / "timing.csv").exists():
        old = pd.read_csv(a.out_dir / "timing.csv")
        out = pd.concat([old[~old.label.isin(out.label)], out], ignore_index=True)
    out.to_csv(a.out_dir / "timing.csv", index=False)
    (a.out_dir / "timing.json").write_text(json.dumps({
        "measured": datetime.now().isoformat(timespec="seconds"), "repeats": a.repeats, "seed": a.seed, "features": cols,
        "n_test_trips": n_trips, "n_test_candidates": n_cand, "logical_cpus": os.cpu_count(), "platform": platform.platform(),
        "note": "fit time = one fit on all 80 training trips; inference = median of repeated predict_proba on the complete student-test set after a warm-up call",
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
