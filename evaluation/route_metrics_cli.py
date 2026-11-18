"""
Adds route metrics (classifier-only: highest-probability candidate per GPS
point) to harness run folders, filling the "Route continuity" column of the
report 3.1 comparison table built by person2/experiments/compare.py.

    python evaluation/route_metrics_cli.py person2/experiments/logs/<run_id> [<run_id> ...]

Needs the run's probability files (proba_student_test.parquet, and
oof_proba_train.parquet for the CV numbers), which are kept outside git, so
run it on the machine that produced the run. It rewrites "route_student_test"
and "route_cv" in that run's metrics.json and saves route_metrics_per_trip.csv.
Definitions: evaluation/route_metrics.py::compute_route_metrics.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from evaluation.route_metrics import route_metrics_for_predictions  # noqa: E402

SOURCE = "evaluation/route_metrics.py, classifier-only (argmax candidate per observation, no HMM)"


def add_route_metrics(run_dir: Path) -> dict:
    metrics_path = run_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text())
    test_proba = run_dir / "proba_student_test.parquet"
    if not test_proba.exists():
        raise FileNotFoundError(f"{test_proba} not found (probability files are not in git; run where the run was made)")
    metrics["route_student_test"] = {**route_metrics_for_predictions(
        pd.read_parquet(test_proba), per_trip_out=run_dir / "route_metrics_per_trip.csv"), "source": SOURCE}
    oof = run_dir / "oof_proba_train.parquet"
    if oof.exists():
        metrics["route_cv"] = {**route_metrics_for_predictions(pd.read_parquet(oof)), "source": SOURCE}
    metrics_path.write_text(json.dumps(metrics, indent=2))
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+", type=Path)
    for run_dir in ap.parse_args().run_dirs:
        m = add_route_metrics(run_dir)
        t = m["route_student_test"]
        print(f"{run_dir.name}: student_test route continuity {t['route_continuity']:.4f} "
              f"(arc-change {t['arc_change_continuity']:.4f}), pointwise {t['pointwise_accuracy']:.4f}")


if __name__ == "__main__":
    main()
