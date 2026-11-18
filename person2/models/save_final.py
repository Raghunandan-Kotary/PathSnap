"""Promote one LOGGED run to the final model that the API loads.

The model is not re-trained or hand-edited here: the file copied is the `model.joblib` that
`run_experiment(save_model=True)` wrote during a logged run (for the tuned model, the run that
`tuning/run_optuna.py` creates). `config.json` records that run's id, parameters, feature order, seed
and the data hash, so every number about the final model traces back to a log in the repo.

    python models/save_final.py --run-dir experiments/logs/<tuned-run-id> [--out-dir models/final]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from person2 import config  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--out-dir", type=Path, default=config.MODELS_DIR / "final")
    a = ap.parse_args()

    src = a.run_dir / "model.joblib"
    if not src.exists():
        sys.exit(f"{src} not found: re-run that experiment with --save-model (tuning/run_optuna.py saves the tuned one)")
    cfg = json.loads((a.run_dir / "config.json").read_text())
    met = json.loads((a.run_dir / "metrics.json").read_text())
    if cfg.get("smoke") and a.out_dir.resolve() == (config.MODELS_DIR / "final").resolve():
        sys.exit("refusing to promote a smoke run (stand-in data) to models/final")

    a.out_dir.mkdir(parents=True, exist_ok=True)
    dst = a.out_dir / "model.joblib"
    shutil.copy2(src, dst)
    (a.out_dir / "config.json").write_text(json.dumps({
        "promoted_at": datetime.now().isoformat(timespec="seconds"), "source_run_id": cfg["run_id"],
        "model": cfg["model"], "family": cfg["family"], "params": cfg["params"], "calibration": cfg["calibration"],
        "threshold": cfg["threshold"], "seed": cfg["seed"], "feature_cols": cfg["feature_cols"],
        "trained_on": f"{cfg['n_train_trips']} training trips only (student_test and evaluator trips not used for fitting)",
        "table_path": cfg["table_path"], "table_sha1": cfg["table_sha1"], "versions": cfg["versions"],
        "model_file_sha256": hashlib.sha256(dst.read_bytes()).hexdigest(),
        "logged_metrics": {"cv_macro_f1_mean": met["cv_macro_f1_mean"], "student_test": {
            k: met["student_test"][k] for k in ("top1_acc", "macro_f1", "log_loss", "infer_ms_per_trip")}},
    }, indent=2), encoding="utf-8")
    print(f"promoted {cfg['run_id']} -> {dst}")


if __name__ == "__main__":
    main()
