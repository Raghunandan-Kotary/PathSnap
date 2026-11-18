"""Run the teammates' models through the shared harness on the frozen table (full folds, no resampling).

    python experiments/run_teammate_models.py --table <candidates_features.parquet> --obs <observations.parquet> \
        --features feat_dist_to_arc feat_arc_length feat_heading_diff feat_speed feat_time_gap [--models ...]

Why: the numbers in the team repo's results JSON/CSV were produced on class-rebalanced random subsamples
(and KNN was scored on a 50,000-row slice of the test candidates), so they are not comparable with each
other or with the real candidate distribution. Here every model is trained on the same folds and scored
on complete validation folds and the complete student-test set. Training time is logged per model.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from person2 import config  # noqa: E402
from person2.experiments.harness import load_table, run_experiment  # noqa: E402
from person2.experiments.models import REGISTRY, TEAMMATE_MODELS  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True, type=Path)
    ap.add_argument("--obs", type=Path, default=None)
    ap.add_argument("--features", nargs="+", default=None)
    ap.add_argument("--models", nargs="+", default=[s.name for s in TEAMMATE_MODELS], choices=sorted(REGISTRY))
    ap.add_argument("--seed", type=int, default=config.GLOBAL_SEED)
    ap.add_argument("--log-dir", type=Path, default=config.EXPERIMENT_LOGS)
    a = ap.parse_args()

    df, obs, cols = load_table(a.table, a.obs, feature_cols=a.features)
    print(f"{len(cols)} features: {cols}\n", flush=True)
    for name in a.models:
        print(f"== {name}", flush=True)
        run_experiment(REGISTRY[name], df, cols, obs, seed=a.seed, role="baseline", table_path=a.table, log_dir=a.log_dir)


if __name__ == "__main__":
    main()
