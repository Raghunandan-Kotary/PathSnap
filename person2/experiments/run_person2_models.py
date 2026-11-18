"""Run Person 2's models through the shared harness, in the order used for the report.

    python experiments/run_person2_models.py --table features/candidates.parquet --obs features/observations.parquet

Runs, each logged under experiments/logs/<run_id>/:
    xgboost                (role=baseline)           initial hyperparameters from experiments/models.py
    gaussian_nb            (role=baseline)
    xgboost + isotonic     (role=calibration_variant) probability-calibration check
    gaussian_nb + isotonic (role=calibration_variant)
Then:  python experiments/compare.py --table ...   builds the table and plots from the logs.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from person2 import config  # noqa: E402
from person2.experiments.harness import load_table, run_experiment  # noqa: E402
from person2.experiments.models import GAUSSIAN_NB, XGBOOST  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True, type=Path)
    ap.add_argument("--obs", type=Path, default=None)
    ap.add_argument("--calibrations", nargs="*", default=["isotonic"], choices=["isotonic", "sigmoid"])
    ap.add_argument("--seed", type=int, default=config.GLOBAL_SEED)
    ap.add_argument("--smoke", action="store_true", help="stand-in data: logs go to experiments/smoke/logs and are tagged smoke")
    ap.add_argument("--log-dir", type=Path, default=None, help="override where run folders are written")
    ap.add_argument("--tag", default="", help="suffix added to every run id, e.g. 'interim'")
    ap.add_argument("--feature-prefix", default=None, help="feature column prefix (auto-detects feat_ or f_)")
    ap.add_argument("--features", nargs="+", default=None, help="use only these feature columns")
    a = ap.parse_args()

    log_dir = a.log_dir or (config.REPO_ROOT / "experiments" / "smoke" / "logs" if a.smoke else config.EXPERIMENT_LOGS)
    df, obs, cols = load_table(a.table, a.obs, feature_cols=a.features, feature_prefix=a.feature_prefix)
    print(f"{len(cols)} features: {cols}\n")
    plan = [(spec, None, "baseline", "") for spec in (XGBOOST, GAUSSIAN_NB)]
    plan += [(spec, cal, "calibration_variant", cal[:3]) for spec in (XGBOOST, GAUSSIAN_NB) for cal in a.calibrations]
    for spec, cal, role, tag in plan:
        print(f"== {spec.name} calibration={cal}")
        full_tag = "_".join(t for t in (a.tag, tag) if t)
        run_experiment(spec, df, cols, obs, seed=a.seed, calibration=cal, tag=full_tag, role=role, smoke=a.smoke,
                       table_path=a.table, log_dir=log_dir)


if __name__ == "__main__":
    main()
