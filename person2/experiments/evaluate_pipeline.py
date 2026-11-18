"""Run the full API pipeline (features -> model probabilities -> Viterbi -> server-side scoring) over every trip
of one partition and log the result. This is exactly the code path the demo uses (api/pipeline.py).

    python person2/experiments/evaluate_pipeline.py --partition student_test
    python person2/experiments/evaluate_pipeline.py --partition evaluator_test     # final evaluation, run once

Data comes from data/raw/<partition>/ (git-ignored). Writes experiments/results/final_pipeline_<partition>.csv and
.json. The evaluator partition is only ever scored here, after every model and decoder choice is final.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from api.config import settings  # noqa: E402
from api.pipeline import pipeline_instance  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--partition", required=True, choices=["student_test", "evaluator_test"])
    ap.add_argument("--out-dir", type=Path, default=ROOT / "person2" / "experiments" / "results")
    a = ap.parse_args()

    part_dir = settings.RAW_DATA_DIR / a.partition
    trips = sorted(d.name for d in part_dir.iterdir() if d.is_dir())
    rows = []
    for t in trips:
        r = pipeline_instance.run_pipeline(t, include_map=False)
        m = r.metrics
        rows.append({"trip_id": t, "execution_mode": r.execution_mode, "observations": m.total_observations,
                     "matched": m.matched_observations, "pointwise_accuracy": m.pointwise_top1_accuracy,
                     "route_continuity": m.route_continuity, "total_latency_ms": r.timing.total_latency_ms})
        print(f"{t}: {r.execution_mode:20s} accuracy {m.pointwise_top1_accuracy:.4f} continuity {m.route_continuity:.4f}", flush=True)
    df = pd.DataFrame(rows)
    summary = {
        "partition": a.partition, "evaluated": datetime.now().isoformat(timespec="seconds"), "n_trips": len(df),
        "execution_modes": df.execution_mode.value_counts().to_dict(),
        "pooled_pointwise_accuracy": float(df.matched.sum() / df.observations.sum()),
        "mean_trip_accuracy": float(df.pointwise_accuracy.mean()),
        "mean_route_continuity": float(df.route_continuity.mean()),
        "model_artifact": str(pipeline_instance.model is not None),
        "definition": "evaluation/route_metrics.py::compute_route_metrics (per-observation reference arc)",
    }
    a.out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out_dir / f"final_pipeline_{a.partition}.csv", index=False)
    (a.out_dir / f"final_pipeline_{a.partition}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
