"""Shared evaluation harness (Person 2). Every teammate trains through `run_experiment`.

For one model + parameter set it:
  1. runs trip-grouped K-fold CV on the 80 training trips (the `fold` column; trips never split),
     fitting all preprocessing inside each fold (no leakage),
  2. refits on all training trips and scores the 10 student-test trips,
  3. computes identical metrics for every model (see experiments/metrics.py),
  4. logs config, seed, library versions, data hash, fit time and inference time, and writes
     per-candidate probabilities (out-of-fold for train, final-model for student_test) with the row
     order of the input table unchanged, ready for the HMM.

The evaluator-testing partition does not exist here and must never be added to the table.
"""
from __future__ import annotations

import hashlib
import json
import platform
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import pandas as pd
import sklearn

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from person2 import config  # noqa: E402
from person2.experiments.calibration import GroupCalibrated  # noqa: E402
from person2.experiments.metrics import candidate_metrics, observation_metrics  # noqa: E402

REQUIRED_COLS = ["trip_id", "obs_idx", "arc_id", "label", "partition", "fold"]
FEATURE_PREFIX = "f_"


@dataclass
class ModelSpec:
    """How to build one model. `build(params, seed)` returns an unfitted sklearn estimator/Pipeline
    exposing fit / predict_proba. Preprocessing (scalers, encoders) must live INSIDE it so the
    harness fits it on training folds only."""

    name: str
    family: str
    build: Callable[[dict, int], object]
    defaults: dict = field(default_factory=dict)


def sha1_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def load_table(table_path: Path, obs_path: Path | None = None, feature_cols: list[str] | None = None,
               feature_prefix: str | None = None, allow_constant: bool = False):
    """Reads the candidate table (parquet/csv) and optional observation table; validates the schema."""
    table_path = Path(table_path)
    df = pd.read_parquet(table_path) if table_path.suffix == ".parquet" else pd.read_csv(table_path, dtype={"trip_id": str})
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"candidate table is missing required columns {missing}; see docs/INTERFACES.md")
    df["trip_id"] = df["trip_id"].astype(str).str.zfill(8)
    parts = set(df.partition.unique())
    if not parts <= {"train", "student_test"}:
        raise ValueError(f"unexpected partition values {parts - {'train', 'student_test'}}: only train and student_test are allowed")
    if (df.groupby("trip_id").partition.nunique() > 1).any():
        raise ValueError("a trip appears in more than one partition")
    tr = df[df.partition == "train"]
    if (tr.groupby("trip_id").fold.nunique() > 1).any() or (tr.fold < 0).any():
        raise ValueError("train trips must each have exactly one fold id >= 0")
    if feature_prefix is None:  # team schema uses `feat_`, the original proposal used `f_`
        feature_prefix = next((p for p in ("feat_", FEATURE_PREFIX) if any(c.startswith(p) for c in df.columns)), FEATURE_PREFIX)
    cols = feature_cols or [c for c in df.columns if c.startswith(feature_prefix)]
    if not cols:
        raise ValueError(f"no feature columns found (expected columns starting with {feature_prefix!r})")
    if df[cols].isna().any().any():
        raise ValueError("feature columns contain NaN; impute upstream (and fit imputers on train only)")
    constant = [c for c in cols if df[c].nunique(dropna=True) <= 1]
    if constant and not allow_constant:
        raise ValueError(f"feature columns are constant and carry no information: {constant}")
    obs = None
    if obs_path is not None and Path(obs_path).exists():
        obs = pd.read_parquet(obs_path) if Path(obs_path).suffix == ".parquet" else pd.read_csv(obs_path, dtype={"trip_id": str})
        obs["trip_id"] = obs["trip_id"].astype(str).str.zfill(8)
    return df, obs, cols


def _fit(est, X, y, groups, calibration: str | None):
    if calibration:
        return GroupCalibrated(est, method=calibration).fit(X, y, groups)
    return est.fit(X, y)


def _predict(model, X) -> np.ndarray:
    return model.predict_proba(X)[:, 1]


def cross_validate(
    spec: ModelSpec,
    train: pd.DataFrame,
    feature_cols: list[str],
    obs: pd.DataFrame | None,
    params: dict,
    *,
    seed: int,
    calibration: str | None = None,
    threshold: float = 0.5,
    on_fold: Callable[[int, list], None] | None = None,
    verbose: bool = False,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Trip-grouped CV on the TRAIN partition only. Returns (per-fold metrics, out-of-fold probabilities).

    `train` must contain training trips only; the student_test partition is never passed in here, so
    hyperparameter search that calls this function cannot see it. `on_fold(step, fold_rows)` is called
    after each fold (Optuna uses it to prune weak trials; it may raise to stop early).
    """
    assert (train.partition == "train").all(), "cross_validate must only ever see training trips"
    X, y = train[feature_cols].to_numpy(dtype="float32"), train["label"].to_numpy()
    groups = train.trip_id.to_numpy()
    fold_rows, oof = [], np.full(len(train), np.nan)
    for k in sorted(train.fold.unique()):
        va = (train.fold == k).to_numpy()
        t0 = time.perf_counter()
        m = _fit(spec.build(params, seed), X[~va], y[~va], groups[~va], calibration)
        fit_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        p = _predict(m, X[va])
        inf_s = time.perf_counter() - t0
        oof[va] = p
        pf = train.loc[va, ["trip_id", "obs_idx", "arc_id", "label"]].assign(proba=p)
        fm = {**candidate_metrics(y[va], p, threshold), **observation_metrics(pf, obs)}
        fm.update(fold=int(k), n_trips=int(train.loc[va, "trip_id"].nunique()), fit_time_s=fit_s, infer_time_s=inf_s)
        fold_rows.append(fm)
        if verbose:
            print(f"  fold {k}: macroF1={fm['macro_f1']:.4f} top1={fm['top1_acc']:.4f} logloss={fm['log_loss']:.4f} (fit {fit_s:.1f}s)", flush=True)
        if on_fold is not None:
            on_fold(len(fold_rows) - 1, fold_rows)
    return pd.DataFrame(fold_rows), oof


def _portable_path(p: Path | None) -> str | None:
    """Repo-relative path when possible, else just the file name: logs get committed to a public repo,
    so they must not embed local folder names."""
    if p is None:
        return None
    p = Path(p).resolve()
    try:
        return p.relative_to(config.REPO_ROOT).as_posix()
    except ValueError:
        return p.name


def run_experiment(
    spec: ModelSpec,
    df: pd.DataFrame,
    feature_cols: list[str],
    obs: pd.DataFrame | None = None,
    params: dict | None = None,
    *,
    seed: int = config.GLOBAL_SEED,
    calibration: str | None = None,
    threshold: float = 0.5,
    tag: str = "",
    role: str = "baseline",
    smoke: bool = False,
    table_path: Path | None = None,
    save_model: bool = False,
    route_metrics_fn: Callable[[pd.DataFrame], dict] | None = None,
    log_dir: Path = config.EXPERIMENT_LOGS,
    verbose: bool = True,
) -> dict:
    params = {**spec.defaults, **(params or {})}
    run_id = f"{datetime.now():%Y%m%d-%H%M%S}_{spec.name}" + (f"_{tag}" if tag else "")
    out_dir = Path(log_dir) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    train = df[df.partition == "train"]
    test = df[df.partition == "student_test"]
    X_tr, y_tr = train[feature_cols].to_numpy(dtype="float32"), train["label"].to_numpy()
    folds = sorted(train.fold.unique())

    # ---------------- trip-grouped cross-validation ----------------
    fold_df, oof = cross_validate(spec, train, feature_cols, obs, params, seed=seed, calibration=calibration,
                                  threshold=threshold, verbose=verbose)
    oof_pred = train[["trip_id", "obs_idx", "arc_id", "label"]].assign(proba=oof)
    cv_pooled = {**candidate_metrics(y_tr, oof, threshold), **observation_metrics(oof_pred, obs)}

    # ---------------- refit on all training trips, score student_test ----------------
    t0 = time.perf_counter()
    final = _fit(spec.build(params, seed), X_tr, y_tr, train.trip_id.to_numpy(), calibration)
    train_time_s = time.perf_counter() - t0
    X_te, y_te = test[feature_cols].to_numpy(dtype="float32"), test["label"].to_numpy()
    _predict(final, X_te[:256])  # warm-up so one-off start-up cost is not billed as inference time
    t0 = time.perf_counter()
    p_te = _predict(final, X_te)
    infer_s = time.perf_counter() - t0
    te_pred = test[["trip_id", "obs_idx", "arc_id", "label"]].assign(proba=p_te)
    test_m = {**candidate_metrics(y_te, p_te, threshold), **observation_metrics(te_pred, obs)}
    test_m.update(train_time_s=train_time_s, infer_time_s=infer_s,
                  infer_ms_per_1k_candidates=1000 * infer_s / max(len(test), 1) * 1000,
                  infer_ms_per_trip=1000 * infer_s / max(test.trip_id.nunique(), 1))

    # route continuity / sequence-level: supplied by Person 3's evaluation code
    route_cv = route_test = {"route_continuity": None, "sequence_score": None}
    if route_metrics_fn is not None:
        route_cv, route_test = route_metrics_fn(oof_pred), route_metrics_fn(te_pred)

    # ---------------- persist ----------------
    key_cols = ["trip_id", "obs_idx", "arc_id"]
    train[key_cols].assign(proba=oof).to_parquet(out_dir / "oof_proba_train.parquet", index=False)
    test[key_cols].assign(proba=p_te).to_parquet(out_dir / "proba_student_test.parquet", index=False)
    fold_df.to_csv(out_dir / "fold_metrics.csv", index=False)
    if save_model:
        joblib.dump({"model": final, "feature_cols": feature_cols, "params": params, "seed": seed,
                     "calibration": calibration, "threshold": threshold}, out_dir / "model.joblib")

    cfg = {
        "run_id": run_id, "timestamp": datetime.now().isoformat(timespec="seconds"), "smoke": smoke, "tag": tag,
        "role": role,  # baseline | calibration_variant | untuned_ref | tuned
        "model": spec.name, "family": spec.family, "params": params, "calibration": calibration,
        "threshold": threshold, "seed": seed, "cv_folds": [int(k) for k in folds],
        "n_train_trips": int(train.trip_id.nunique()), "n_test_trips": int(test.trip_id.nunique()),
        "n_train_candidates": int(len(train)), "n_test_candidates": int(len(test)),
        "feature_cols": feature_cols, "label_stats": _label_stats(df),
        "table_path": _portable_path(table_path),
        "table_sha1": sha1_file(Path(table_path)) if table_path and Path(table_path).exists() else None,
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                     "sklearn": sklearn.__version__},
    }
    for lib in ("xgboost", "lightgbm"):
        try:
            cfg["versions"][lib] = __import__(lib).__version__
        except ImportError:
            pass
    metrics = {
        "cv_per_fold_macro_f1": fold_df.macro_f1.round(6).tolist(),
        "cv_macro_f1_mean": float(fold_df.macro_f1.mean()), "cv_macro_f1_std": float(fold_df.macro_f1.std(ddof=1)),
        "cv_top1_mean": float(fold_df.top1_acc.mean()), "cv_log_loss_mean": float(fold_df.log_loss.mean()),
        "cv_pooled": cv_pooled, "student_test": test_m, "route_cv": route_cv, "route_student_test": route_test,
    }
    with open(out_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2, default=_json_default)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=_json_default)
    _append_master_log(Path(log_dir) / "runs.csv", cfg, metrics)
    if verbose:
        print(f"[{run_id}] CV macroF1 {metrics['cv_macro_f1_mean']:.4f}±{metrics['cv_macro_f1_std']:.4f} | "
              f"test top1 {test_m['top1_acc']:.4f} macroF1 {test_m['macro_f1']:.4f} logloss {test_m['log_loss']:.4f} | "
              f"train {train_time_s:.1f}s infer {infer_s*1000:.0f}ms")
    return {"config": cfg, "metrics": metrics, "run_dir": out_dir}


def _label_stats(df: pd.DataFrame) -> dict:
    """Class balance and label ambiguity per partition (logged with every run)."""
    out = {}
    for part, g in df.groupby("partition"):
        per_obs = g.groupby(["trip_id", "obs_idx"])["label"].agg(["sum", "size"])
        out[part] = {
            "candidates": int(len(g)), "positive_rate": float(g.label.mean()),
            "observations_with_candidates": int(len(per_obs)),
            "mean_candidates_per_obs": float(per_obs["size"].mean()),
            "mean_positives_per_obs": float(per_obs["sum"].mean()),
            "obs_with_zero_positive": float((per_obs["sum"] == 0).mean()),
            "obs_with_one_positive": float((per_obs["sum"] == 1).mean()),
            "obs_with_2plus_positive": float((per_obs["sum"] >= 2).mean()),
        }
    return out


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def _append_master_log(path: Path, cfg: dict, m: dict) -> None:
    t, cv = m["student_test"], m["cv_pooled"]
    row = {
        "run_id": cfg["run_id"], "timestamp": cfg["timestamp"], "smoke": cfg["smoke"], "model": cfg["model"],
        "family": cfg["family"], "tag": cfg["tag"], "role": cfg["role"], "calibration": cfg["calibration"],
        "seed": cfg["seed"],
        "key_config": json.dumps(cfg["params"], sort_keys=True, default=_json_default),
        "n_train_trips": cfg["n_train_trips"], "n_test_trips": cfg["n_test_trips"], "n_features": len(cfg["feature_cols"]),
        "cv_macro_f1_mean": m["cv_macro_f1_mean"], "cv_macro_f1_std": m["cv_macro_f1_std"],
        "cv_top1_mean": m["cv_top1_mean"], "cv_log_loss_mean": m["cv_log_loss_mean"],
        "cv_pooled_macro_f1": cv["macro_f1"], "cv_pooled_log_loss": cv["log_loss"],
        "test_top1": t["top1_acc"], "test_top1_hit_only": t["top1_acc_hit_only"], "test_macro_f1": t["macro_f1"],
        "test_macro_precision": t["macro_precision"], "test_macro_recall": t["macro_recall"],
        "test_log_loss": t["log_loss"], "test_brier": t["brier"], "test_ece": t["ece"], "test_roc_auc": t["roc_auc"],
        "test_candidate_miss_rate": t["candidate_miss_rate"], "miss_rate_scope": t["miss_rate_scope"],
        "test_route_continuity": m["route_student_test"].get("route_continuity"),
        "train_time_s": t["train_time_s"], "infer_ms_per_1k_candidates": t["infer_ms_per_1k_candidates"],
        "infer_ms_per_trip": t["infer_ms_per_trip"], "run_dir": cfg["run_id"],
    }
    new = pd.DataFrame([row])
    if path.exists():
        new = pd.concat([pd.read_csv(path), new], ignore_index=True)
    new.to_csv(path, index=False)
