"""Bayesian hyperparameter optimisation (Optuna TPE) with trip-grouped CV on the 80 training trips.

The search sees ONLY the training partition (`cross_validate` asserts this). The student-test trips
are used afterwards, once, to compare the tuned model against the untuned one. The evaluator
partition does not exist in this code base.

    python tuning/run_optuna.py --model xgboost --table features/candidates.parquet \
        --obs features/observations.parquet --n-trials 60

Outputs in --out-dir (default tuning/studies/<model>_<objective>/):
    study.db               Optuna SQLite storage (resumable; not committed)
    trials.csv             FULL trial log: params, value, state, duration (the search record)
    best_params.json       best trial
    study_config.json      search space, sampler, pruner, seed, trial counts, wall time, versions
    optimization_history.png, param_importance.png
    tuned_vs_untuned.csv / .md   same metrics and partitions for the default and the tuned model
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import optuna  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from person2 import config  # noqa: E402
from person2.experiments.harness import _portable_path, cross_validate, load_table, run_experiment, sha1_file  # noqa: E402
from person2.experiments.models import REGISTRY  # noqa: E402

SPACE_FILE = Path(__file__).resolve().parent / "search_space.json"


def suggest(trial: optuna.Trial, space: dict) -> dict:
    params = {}
    for name, s in space.items():
        if s["type"] == "int":
            params[name] = trial.suggest_int(name, s["low"], s["high"], step=s.get("step", 1), log=s.get("log", False))
        else:
            params[name] = trial.suggest_float(name, s["low"], s["high"], log=s.get("log", False))
    return params


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="xgboost", choices=sorted(REGISTRY))
    ap.add_argument("--table", required=True, type=Path)
    ap.add_argument("--obs", type=Path, default=None)
    ap.add_argument("--features", nargs="+", default=None, help="use only these feature columns (default: all feat_*/f_* columns)")
    ap.add_argument("--objective", default="cv_log_loss", choices=["cv_macro_f1", "cv_top1", "cv_log_loss"])
    ap.add_argument("--n-trials", type=int, default=60)
    ap.add_argument("--n-startup-trials", type=int, default=10, help="random trials before TPE starts modelling")
    ap.add_argument("--timeout", type=float, default=None, help="optional wall-clock limit in seconds")
    ap.add_argument("--calibrate", choices=["isotonic", "sigmoid"], default=None)
    ap.add_argument("--seed", type=int, default=config.GLOBAL_SEED)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--log-dir", type=Path, default=config.EXPERIMENT_LOGS, help="where the tuned/untuned runs are logged")
    ap.add_argument("--smoke", action="store_true", help="stand-in data: tag runs smoke=True")
    ap.add_argument("--no-finalize", action="store_true", help="skip the tuned-vs-untuned comparison")
    a = ap.parse_args()

    spaces = json.loads(SPACE_FILE.read_text(encoding="utf-8"))
    space, obj_def = spaces[a.model], spaces["objectives"][a.objective]
    metric, direction = obj_def["metric"], obj_def["direction"]
    out = a.out_dir or (config.TUNING_DIR / "studies" / f"{a.model}_{a.objective}")
    out.mkdir(parents=True, exist_ok=True)

    df, obs, cols = load_table(a.table, a.obs, feature_cols=a.features)
    train = df[df.partition == "train"]  # the search never sees student_test
    spec = REGISTRY[a.model]
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: optuna.Trial) -> float:
        params = suggest(trial, space)

        def report_and_prune(step: int, rows: list) -> None:
            trial.report(float(np.mean([r[metric] for r in rows])), step)
            if trial.should_prune():
                raise optuna.TrialPruned()

        fold_df, _ = cross_validate(spec, train, cols, obs, {**spec.defaults, **params}, seed=a.seed,
                                    calibration=a.calibrate, on_fold=report_and_prune)
        trial.set_user_attr("cv_macro_f1_mean", float(fold_df.macro_f1.mean()))
        trial.set_user_attr("cv_macro_f1_std", float(fold_df.macro_f1.std(ddof=1)))
        trial.set_user_attr("cv_top1_mean", float(fold_df.top1_acc.mean()))
        trial.set_user_attr("cv_log_loss_mean", float(fold_df.log_loss.mean()))
        trial.set_user_attr("fit_time_total_s", float(fold_df.fit_time_s.sum()))
        return float(fold_df[metric].mean())

    sampler = optuna.samplers.TPESampler(seed=a.seed, n_startup_trials=a.n_startup_trials)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)
    study = optuna.create_study(study_name=f"{a.model}_{a.objective}", storage=f"sqlite:///{(out / 'study.db').as_posix()}",
                                sampler=sampler, pruner=pruner, direction=direction, load_if_exists=True)
    already = len(study.trials)
    t0 = time.perf_counter()
    study.optimize(objective, n_trials=max(a.n_trials - already, 0), timeout=a.timeout, gc_after_trial=True)
    wall = time.perf_counter() - t0

    trials = study.trials_dataframe()
    trials.to_csv(out / "trials.csv", index=False)
    states = trials["state"].value_counts().to_dict()
    best = study.best_trial
    (out / "best_params.json").write_text(json.dumps({
        "model": a.model, "objective": a.objective, "direction": direction, "best_value": best.value,
        "best_trial_number": best.number, "params": best.params, "user_attrs": best.user_attrs}, indent=2), encoding="utf-8")
    (out / "study_config.json").write_text(json.dumps({
        "created": datetime.now().isoformat(timespec="seconds"), "smoke": a.smoke, "model": a.model,
        "algorithm": "Bayesian optimisation: Tree-structured Parzen Estimator (optuna.samplers.TPESampler)",
        "pruner": "MedianPruner(n_startup_trials=5, n_warmup_steps=1), reported after each CV fold",
        "objective": a.objective, "metric": metric, "direction": direction, "objective_why": obj_def["why"],
        "calibration": a.calibrate, "seed": a.seed, "n_startup_random_trials": a.n_startup_trials,
        "n_trials_requested": a.n_trials, "trial_states": states, "optimisation_wall_time_s_this_invocation": wall,
        "cv": "trip-grouped folds on the training partition only", "search_space": space,
        "features": cols, "table_path": _portable_path(a.table), "table_sha1": sha1_file(a.table), "n_train_trips": int(train.trip_id.nunique()),
        "versions": {"optuna": optuna.__version__}}, indent=2), encoding="utf-8")

    # ---- figures ----
    fig, ax = plt.subplots(figsize=(8, 4.5))
    done = trials[trials.state == "COMPLETE"]
    pruned = trials[trials.state == "PRUNED"]
    ax.scatter(done.number, done.value, s=22, label=f"complete ({len(done)})")
    if len(pruned):
        ax.scatter(pruned.number, pruned.value, s=22, marker="x", color="grey", label=f"pruned ({len(pruned)})")
    run_best = done.value.cummax() if direction == "maximize" else done.value.cummin()
    ax.plot(done.number, run_best, color="C3", label="best so far")
    ax.set(xlabel="trial", ylabel=f"{a.objective} (mean over folds)", title=f"{a.model}: Bayesian optimisation history")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "optimization_history.png", dpi=130)
    plt.close(fig)
    try:
        imp = optuna.importance.get_param_importances(study)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.barh(list(imp)[::-1], list(imp.values())[::-1])
        ax.set(xlabel="importance (fANOVA)", title=f"{a.model}: hyperparameter importance")
        fig.tight_layout()
        fig.savefig(out / "param_importance.png", dpi=130)
        plt.close(fig)
    except Exception as e:  # importance needs >= 2 completed trials
        print("param importance skipped:", e)

    print(f"\n{len(done)} complete, {len(pruned)} pruned; best {a.objective} = {best.value:.5f} (trial {best.number}); "
          f"{wall:.0f}s this invocation\nbest params: {best.params}")

    # ---- tuned vs untuned: same table, same folds, same seed, same metrics ----
    if not a.no_finalize:
        rows = {}
        for label, role, p in (("untuned", "untuned_ref", {}), ("tuned", "tuned", best.params)):
            res = run_experiment(spec, df, cols, obs, p, seed=a.seed, calibration=a.calibrate, tag=label, role=role,
                                 smoke=a.smoke, table_path=a.table, save_model=(label == "tuned"),
                                 log_dir=a.log_dir, verbose=False)
            m, t = res["metrics"], res["metrics"]["student_test"]
            rows[label] = {
                "run_id": res["config"]["run_id"], "cv_macro_f1_mean": m["cv_macro_f1_mean"], "cv_macro_f1_std": m["cv_macro_f1_std"],
                "student_test_top1": t["top1_acc"], "student_test_macro_f1": t["macro_f1"], "student_test_log_loss": t["log_loss"],
                "inference_ms_per_trip": t["infer_ms_per_trip"], "inference_ms_per_1k_candidates": t["infer_ms_per_1k_candidates"],
                "train_time_s": t["train_time_s"]}
        cmp_df = pd.DataFrame(rows).T
        delta = (cmp_df.loc["tuned"].drop("run_id").astype(float) - cmp_df.loc["untuned"].drop("run_id").astype(float))
        cmp_df.loc["difference (tuned - untuned)"] = delta
        cmp_df.to_csv(out / "tuned_vs_untuned.csv")
        (out / "tuned_vs_untuned.md").write_text(cmp_df.to_markdown(floatfmt=".4f") if _has_tabulate() else cmp_df.to_string(), encoding="utf-8")
        print("\n" + cmp_df.drop(columns="run_id").astype(float, errors="ignore").round(4).to_string())


def _has_tabulate() -> bool:
    try:
        import tabulate  # noqa: F401
        return True
    except ImportError:
        return False


if __name__ == "__main__":
    main()
