"""Consolidated model comparison (report sections 3.1 and 3.3) built ONLY from logged runs.

Reads every `experiments/logs/<run_id>/{config.json,metrics.json,fold_metrics.csv}` (so run folders
copied in from teammates merge without conflicts), keeps one run per model label, and writes:

    comparison_table.csv / .md     the report's table 3.1
    all_runs.csv                   every run found, for audit
    cv_macro_f1.png                CV macro F1 per model (mean, std, per-fold points)
    student_test_top1_macro_f1.png student-test Top-1 and macro F1
    log_loss.png                   CV and student-test log loss
    confusion_matrices.png         student-test confusion matrix per model
    reliability.png                calibration curves (needs --table)

    python experiments/compare.py [--log-dir experiments/logs] [--out-dir experiments/results] [--table T.parquet]
Smoke runs (stand-in data) are excluded unless --include-smoke, and then go to a separate folder name.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from person2 import config  # noqa: E402

NA_ROUTE = "N/A (needs road graph; computed by the decoder metrics)"


def label_of(cfg: dict) -> str:
    return cfg["model"] + (f"+{cfg['calibration']}" if cfg.get("calibration") else "")


def collect(log_dir: Path) -> pd.DataFrame:
    rows = []
    for d in sorted(Path(log_dir).glob("*/")):
        if not (d / "config.json").exists() or not (d / "metrics.json").exists():
            continue
        cfg, m = json.loads((d / "config.json").read_text()), json.loads((d / "metrics.json").read_text())
        t = m["student_test"]
        rows.append({
            "run_id": cfg["run_id"], "timestamp": cfg["timestamp"], "smoke": cfg.get("smoke", False),
            "role": cfg.get("role", "baseline"), "label": label_of(cfg),
            "feature_set": "v2" if any(c.startswith("f2_") for c in cfg["feature_cols"]) else "v1", "model": cfg["model"], "family": cfg["family"],
            "calibration": cfg.get("calibration"), "key_config": json.dumps(cfg["params"], sort_keys=True),
            "cv_macro_f1_mean": m["cv_macro_f1_mean"], "cv_macro_f1_std": m["cv_macro_f1_std"],
            "cv_fold_macro_f1": m["cv_per_fold_macro_f1"], "cv_log_loss_mean": m["cv_log_loss_mean"],
            "test_top1": t["top1_acc"], "test_macro_f1": t["macro_f1"], "test_macro_precision": t["macro_precision"],
            "test_macro_recall": t["macro_recall"], "test_log_loss": t["log_loss"], "test_ece": t["ece"],
            "test_miss_rate": t["candidate_miss_rate"], "route_continuity": m["route_student_test"].get("route_continuity"),
            "infer_ms_per_trip": t["infer_ms_per_trip"], "infer_ms_per_1k": t["infer_ms_per_1k_candidates"],
            "train_time_s": t["train_time_s"], "tn": t["tn"], "fp": t["fp"], "fn": t["fn"], "tp": t["tp"],
            "table_sha1": cfg.get("table_sha1"), "run_dir": str(d),
        })
    return pd.DataFrame(rows)


def pick_runs(runs: pd.DataFrame, roles: list[str]) -> pd.DataFrame:
    sel = runs[runs.role.isin(roles)].sort_values("timestamp")
    dup = sel[sel.duplicated("label", keep="last")]
    if len(dup):
        print("note: several runs share a label; keeping the latest of:", sorted(dup.label.unique()))
    return sel.drop_duplicates("label", keep="last").reset_index(drop=True)


def fmt_config(js: str) -> str:
    p = json.loads(js)
    return ", ".join(f"{k}={v:g}" if isinstance(v, float) else f"{k}={v}" for k, v in p.items())


def apply_timing(sel: pd.DataFrame, results_dir: Path) -> tuple[pd.DataFrame, bool]:
    """Replace noisy per-run timings by the consistent ones from experiments/time_models.py, if present."""
    f = Path(results_dir) / "timing.csv"
    if not f.exists():
        return sel, False
    t = pd.read_csv(f).set_index("label")
    sel = sel.copy()
    for i, lab in enumerate(sel["label"]):
        if lab in t.index:
            sel.loc[sel.index[i], "infer_ms_per_trip"] = t.loc[lab, "infer_ms_per_trip"]
            sel.loc[sel.index[i], "infer_ms_per_1k"] = t.loc[lab, "infer_ms_per_1k_candidates"]
            sel.loc[sel.index[i], "train_time_s"] = t.loc[lab, "fit_time_s"]
    return sel, True


def to_markdown(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines) + "\n"


def report_table(sel: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "Model": sel.label, "Algorithm family": sel.family, "Key configuration": sel.key_config.map(fmt_config),
        "CV macro F1 (mean ± std over folds)": [f"{m:.4f} ± {s:.4f}" for m, s in zip(sel.cv_macro_f1_mean, sel.cv_macro_f1_std)],
        "Top-1 accuracy (student test)": sel.test_top1.map("{:.4f}".format),
        "Route continuity": sel.route_continuity.map(lambda v: NA_ROUTE if v is None or pd.isna(v) else f"{v:.4f}"),
        "Log loss (student test)": sel.test_log_loss.map("{:.4f}".format),
        "Macro F1 (student test)": sel.test_macro_f1.map("{:.4f}".format),
        "Inference time (ms per trip / per 1k candidates)": [f"{a:.1f} / {b:.2f}" for a, b in zip(sel.infer_ms_per_trip, sel.infer_ms_per_1k)],
    })


def plots(sel: pd.DataFrame, out: Path, table: Path | None) -> None:
    labels = sel.label.tolist()
    colors = {l: f"C{i}" for i, l in enumerate(labels)}
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(max(6, 1.6 * len(labels) + 2), 4.5))
    ax.bar(x, sel.cv_macro_f1_mean, yerr=sel.cv_macro_f1_std, capsize=5, color=[colors[l] for l in labels], alpha=0.8)
    for i, folds in enumerate(sel.cv_fold_macro_f1):
        ax.scatter(np.full(len(folds), i), folds, color="black", s=12, zorder=3)
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.set(ylabel="CV macro F1 (trip-grouped folds)", title="Cross-validation macro F1 (bar = mean, whisker = std, dots = folds)")
    fig.tight_layout()
    fig.savefig(out / "cv_macro_f1.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(max(6, 1.8 * len(labels) + 2), 4.5))
    w = 0.38
    ax.bar(x - w / 2, sel.test_top1, w, label="Top-1 accuracy")
    ax.bar(x + w / 2, sel.test_macro_f1, w, label="Macro F1")
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.set(ylabel="score (student-test, 10 trips)", ylim=(0, 1), title="Student-test Top-1 accuracy and macro F1")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "student_test_top1_macro_f1.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(max(6, 1.8 * len(labels) + 2), 4.5))
    ax.bar(x - w / 2, sel.cv_log_loss_mean, w, label="CV mean")
    ax.bar(x + w / 2, sel.test_log_loss, w, label="student test")
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.set(ylabel="log loss (lower is better)", title="Log loss: probability quality for the HMM")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "log_loss.png", dpi=130)
    plt.close(fig)

    n = len(labels)
    fig, axes = plt.subplots(1, n, figsize=(3.4 * n, 3.6), squeeze=False)
    for ax, (_, r) in zip(axes[0], sel.iterrows()):
        cm = np.array([[r.tn, r.fp], [r.fn, r.tp]])
        ax.imshow(cm / cm.sum(axis=1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", color="black", fontsize=9)
        ax.set_xticks([0, 1], ["pred 0", "pred 1"])
        ax.set_yticks([0, 1], ["true 0", "true 1"])
        ax.set_title(r.label, fontsize=10)
    fig.suptitle("Student-test confusion matrices (colour = row-normalised, text = counts, threshold 0.5)")
    fig.tight_layout()
    fig.savefig(out / "confusion_matrices.png", dpi=130)
    plt.close(fig)

    if table is not None:
        from person2.experiments.harness import load_table

        df, _, _ = load_table(table)
        test = df[df.partition == "student_test"][["trip_id", "obs_idx", "arc_id", "label"]]
        fig, ax = plt.subplots(figsize=(5.5, 5.5))
        ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="perfect")
        for _, r in sel.iterrows():
            p = pd.read_parquet(Path(r.run_dir) / "proba_student_test.parquet").merge(test, on=["trip_id", "obs_idx", "arc_id"])
            bins = np.minimum((p.proba.to_numpy() * 10).astype(int), 9)
            g = pd.DataFrame({"b": bins, "p": p.proba, "y": p.label}).groupby("b").agg(p=("p", "mean"), y=("y", "mean"))
            ax.plot(g.p, g.y, marker="o", label=f"{r.label} (ECE {r.test_ece:.3f})")
        ax.set(xlabel="mean predicted probability", ylabel="observed positive rate", title="Reliability (student test)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "reliability.png", dpi=130)
        plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", type=Path, default=config.EXPERIMENT_LOGS)
    ap.add_argument("--out-dir", type=Path, default=config.REPO_ROOT / "experiments" / "results")
    ap.add_argument("--table", type=Path, default=None, help="candidate table (for reliability curves)")
    ap.add_argument("--roles", nargs="+", default=["baseline"],
                    help="which run roles form the comparison table (default: baseline). Add calibration_variant to see them.")
    ap.add_argument("--include-smoke", action="store_true")
    ap.add_argument("--feature-set", choices=["v1", "v2", "any"], default="any",
                    help="v1 = the 5 frozen features, v2 = frozen + context features (experiments/context_features.py)")
    a = ap.parse_args()

    runs = collect(a.log_dir)
    if runs.empty:
        sys.exit(f"no runs found in {a.log_dir}")
    if not a.include_smoke:
        runs = runs[~runs.smoke]
        if runs.empty:
            sys.exit("only smoke runs found; pass --include-smoke to inspect them (never report them)")
    if a.feature_set != "any":
        runs = runs[runs.feature_set == a.feature_set]
    a.out_dir.mkdir(parents=True, exist_ok=True)
    runs.drop(columns=["cv_fold_macro_f1", "run_dir"]).to_csv(a.out_dir / "all_runs.csv", index=False)  # run_dir is a local absolute path
    sel = pick_runs(runs, a.roles)
    sel, timed = apply_timing(sel, a.out_dir)
    tbl = report_table(sel)
    if timed:
        tbl = tbl.rename(columns={"Inference time (ms per trip / per 1k candidates)":
                                  "Inference time (median of 5 repeats; ms per trip / per 1k candidates)"})
    tbl.to_csv(a.out_dir / "comparison_table.csv", index=False)
    (a.out_dir / "comparison_table.md").write_text(to_markdown(tbl), encoding="utf-8")
    plots(sel, a.out_dir, a.table)
    hashes = runs.table_sha1.dropna().unique()
    if len(hashes) > 1:
        print(f"WARNING: runs were trained on {len(hashes)} different feature tables; the comparison is only valid for one.")
    print(f"{len(sel)} models in comparison -> {a.out_dir}")
    print(tbl[["Model", "CV macro F1 (mean ± std over folds)", "Top-1 accuracy (student test)", "Macro F1 (student test)", "Log loss (student test)"]].to_string(index=False))


if __name__ == "__main__":
    main()
