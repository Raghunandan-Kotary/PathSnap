"""Train + evaluate one model through the shared harness.

    python experiments/run_model.py --model xgboost --table features/candidates.parquet \
        --obs features/observations.parquet
    python experiments/run_model.py --model gaussian_nb --calibrate isotonic --tag iso ...
    python experiments/run_model.py --model xgboost --smoke \
        --table experiments/smoke/standin_candidates.parquet --obs experiments/smoke/standin_observations.parquet
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from person2 import config  # noqa: E402
from person2.experiments.harness import load_table, run_experiment  # noqa: E402
from person2.experiments.models import REGISTRY  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(REGISTRY))
    ap.add_argument("--table", required=True, type=Path, help="candidate/feature table (parquet or csv)")
    ap.add_argument("--obs", type=Path, default=None, help="observation table (for candidate miss rate)")
    ap.add_argument("--params", type=json.loads, default={}, help='JSON overrides, e.g. \'{"max_depth": 4}\'')
    ap.add_argument("--calibrate", choices=["isotonic", "sigmoid"], default=None)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=config.GLOBAL_SEED)
    ap.add_argument("--feature-prefix", default=None, help="feature column prefix (auto-detects feat_ or f_)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--features", nargs="+", default=None, help="use only these feature columns (default: all feat_*/f_* columns)")
    ap.add_argument("--role", default="baseline", choices=["baseline", "calibration_variant", "untuned_ref", "tuned", "ablation"],
                    help="how the comparison table should treat this run")
    ap.add_argument("--smoke", action="store_true", help="mark run as smoke test (stand-in data; excluded from the report)")
    ap.add_argument("--save-model", action="store_true")
    ap.add_argument("--log-dir", type=Path, default=config.EXPERIMENT_LOGS)
    a = ap.parse_args()

    df, obs, cols = load_table(a.table, a.obs, feature_cols=a.features, feature_prefix=a.feature_prefix)
    run_experiment(REGISTRY[a.model], df, cols, obs, a.params, seed=a.seed, calibration=a.calibrate,
                   threshold=a.threshold, tag=a.tag, role=a.role, smoke=a.smoke, table_path=a.table,
                   save_model=a.save_model, log_dir=a.log_dir)


if __name__ == "__main__":
    main()
