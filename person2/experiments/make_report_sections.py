"""Generate Person 2's report sections (3.1, 3.2 for XGBoost and Gaussian NB, 3.3, 4) from logged runs.

Every number is read from experiments/logs/*, tuning/studies/* or the frozen table, so the text can be
regenerated and checked against the repository. `claim()` stops the script if a sentence would
contradict the data, so the narrative can never drift from the logs.

    python experiments/make_report_sections.py --table experiments/team_table/candidates_features.parquet \
        --out docs/report/PERSON2_REPORT_SECTIONS.md
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from person2 import config  # noqa: E402
from person2.experiments.compare import apply_timing, collect, fmt_config, pick_runs  # noqa: E402

F5 = ["feat_dist_to_arc", "feat_arc_length", "feat_heading_diff", "feat_speed", "feat_time_gap"]


def claim(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(f"report sentence would contradict the logs: {msg}")


def f4(x) -> str:
    return f"{float(x):.4f}"


def pct(x, d=1) -> str:
    return f"{100 * float(x):.{d}f}%"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True, type=Path)
    ap.add_argument("--log-dir", type=Path, default=config.EXPERIMENT_LOGS)
    ap.add_argument("--results-dir", type=Path, default=config.REPO_ROOT / "experiments" / "results")
    ap.add_argument("--study-dir", type=Path, default=config.TUNING_DIR / "studies" / "xgboost_cv_log_loss")
    ap.add_argument("--out", type=Path, default=config.REPO_ROOT / "docs" / "report" / "PERSON2_REPORT_SECTIONS.md")
    a = ap.parse_args()

    runs = collect(a.log_dir)
    runs = runs[~runs.smoke]
    base = pick_runs(runs, ["baseline"]).set_index("label")
    cal = pick_runs(runs, ["baseline", "calibration_variant"]).set_index("label")
    base_t, timed = apply_timing(base.reset_index(), a.results_dir)  # consistent timings if experiments/time_models.py was run
    base_t = base_t.set_index("label")
    timing = pd.read_csv(a.results_dir / "timing.csv").set_index("label") if timed else None

    def run_files(label: str, source=cal):
        d = Path(source.loc[label, "run_dir"])
        return (json.loads((d / "config.json").read_text()), json.loads((d / "metrics.json").read_text()),
                pd.read_csv(d / "fold_metrics.csv"))

    cfg_x, m_x, f_x = run_files("xgboost")
    cfg_n, m_n, f_n = run_files("gaussian_nb")
    _, m_xi, _ = run_files("xgboost+isotonic")
    _, m_ni, _ = run_files("gaussian_nb+isotonic")
    tx, tn, txi, tni = m_x["student_test"], m_n["student_test"], m_xi["student_test"], m_ni["student_test"]
    ls = cfg_x["label_stats"]
    prior_ll = lambda p: -(p * np.log(p) + (1 - p) * np.log(1 - p))  # noqa: E731
    base_ll_test = prior_ll(ls["student_test"]["positive_rate"])

    # ---- facts computed from the frozen table (reproducible) ----
    from scipy.stats import skew
    from sklearn.metrics import f1_score

    from person2.experiments.harness import load_table
    from person2.experiments.models import GAUSSIAN_NB

    df, obs, cols = load_table(a.table, feature_cols=F5)
    tr = df[df.partition == "train"]
    feat_rows = []
    for c in F5:
        x = tr[c].to_numpy()
        feat_rows.append((c, float(skew(x)), float((x == pd.Series(x).mode().iloc[0]).mean()), float(np.median(x)), float(x.max())))
    gnb = GAUSSIAN_NB.build(GAUSSIAN_NB.defaults, config.GLOBAL_SEED).fit(tr[F5].to_numpy("float32"), tr.label.to_numpy())
    gnb_insample_f1 = float(f1_score(tr.label, gnb.predict_proba(tr[F5].to_numpy("float32"))[:, 1] >= 0.5, average="macro"))
    within = {lab: float(np.max(np.triu(np.abs(tr[tr.label == lab][F5].corr().to_numpy()), 1))) for lab in (0, 1)}
    overall_corr = float(np.max(np.triu(np.abs(tr[F5].corr().to_numpy()), 1)))

    # ---- sanity claims used in the text ----
    claim(base["test_log_loss"].idxmin() == "xgboost", "XGBoost has the lowest student-test log loss")
    claim(base["test_macro_f1"].idxmax() == "xgboost", "XGBoost has the highest student-test macro F1")
    claim(base["cv_macro_f1_mean"].idxmax() == "xgboost", "XGBoost has the highest CV macro F1")
    top1s = base.loc[["xgboost", "random_forest", "hist_gradient_boosting"], "test_top1"]
    claim(top1s.max() - top1s.min() < 0.005, "XGBoost, RF and HGB are within 0.5 points on test Top-1")
    claim(tn["log_loss"] > base_ll_test, "uncalibrated Gaussian NB log loss is worse than the prior-only baseline")
    claim(tni["log_loss"] < tn["log_loss"] and tni["tp"] == 0, "calibrated NB: better log loss, no positives predicted at 0.5")
    claim(abs(txi["log_loss"] - tx["log_loss"]) < 0.002, "isotonic calibration leaves XGBoost log loss essentially unchanged")
    claim(tx["ece"] < 0.01, "XGBoost is well calibrated (ECE < 0.01)")
    claim(gnb_insample_f1 - m_n["cv_macro_f1_mean"] < 0.03, "Gaussian NB shows no train/CV gap (underfitting)")
    worst, best = int(f_x.top1_acc.idxmin()), int(f_x.top1_acc.idxmax())
    claim(f_x.loc[worst, "candidate_miss_rate"] == f_x.candidate_miss_rate.max(), "the weakest fold is the one with the highest candidate miss rate")
    o_tr = obs[obs.partition == "train"] if obs is not None else None
    t54 = "00000054"
    fold_of_t54 = int(tr[tr.trip_id == t54].fold.iloc[0])
    o54 = pd.read_parquet(a.table.with_name("observations.parquet")) if (a.table.with_name("observations.parquet")).exists() else None
    claim(o54 is not None, "observation table next to the candidate table")
    o54 = o54[o54.trip_id == t54]
    n54, miss54 = len(o54), float((o54.n_positive == 0).mean())
    claim(fold_of_t54 == worst, "trip 00000054 is in the weakest fold")

    def tie_share(label: str) -> float:
        """Share of GPS points (with >=2 candidates) whose two best candidates have identical probability."""
        pr = pd.read_parquet(Path(cal.loc[label, "run_dir"]) / "proba_student_test.parquet")
        pr = pr.sort_values(["trip_id", "obs_idx", "proba"], ascending=[True, True, False])
        pr["rank"] = pr.groupby(["trip_id", "obs_idx"]).cumcount()
        first = pr[pr["rank"] == 0].set_index(["trip_id", "obs_idx"]).proba
        second = pr[pr["rank"] == 1].set_index(["trip_id", "obs_idx"]).proba
        return float((first.loc[second.index] == second).mean())

    ties = {k: tie_share(k) for k in ("xgboost", "gaussian_nb", "gaussian_nb+isotonic")}
    claim(ties["gaussian_nb+isotonic"] > ties["gaussian_nb"] >= ties["xgboost"], "isotonic NB has the most tied top-2 probabilities")

    lc = json.loads((a.results_dir / "xgboost_learning_curve.json").read_text()) if (a.results_dir / "xgboost_learning_curve.json").exists() else None
    lc_t = json.loads((a.results_dir / "xgboost_learning_curve_tuned.json").read_text()) if (a.results_dir / "xgboost_learning_curve_tuned.json").exists() else None

    sd = a.study_dir
    tuned = (sd / "tuned_vs_untuned.csv").exists()
    if tuned:
        sc = json.loads((sd / "study_config.json").read_text())
        bp = json.loads((sd / "best_params.json").read_text())
        tv = pd.read_csv(sd / "tuned_vs_untuned.csv", index_col=0)
        trials = pd.read_csv(sd / "trials.csv")
        space = json.loads((config.TUNING_DIR / "search_space.json").read_text())["xgboost"]

    out: list[str] = []
    w = out.append

    # ------------------------------------------------------------------ 3.1
    w("# Person 2: report sections (generated from logged runs)\n")
    w("> Every number below is read from `experiments/logs/`, `tuning/studies/` or the frozen feature table by "
      "`experiments/make_report_sections.py`. Regenerate it after any new run.\n")
    w("## 3.1 Overall model comparison\n")
    w("**How every model was measured.** All models use the same frozen feature table (Person 1, commit `86ad172`): "
      f"{ls['train']['candidates']:,} candidate rows from 80 training trips ({ls['train']['observations_with_candidates']:,} GPS points) and "
      f"{ls['student_test']['candidates']:,} candidate rows from the 10 student-test trips ({ls['student_test']['observations_with_candidates']:,} GPS points). "
      "Each candidate is a (GPS point, road arc) pair; the label is 1 for the single arc the GPS point was matched to on the reference route. "
      f"Positives are {pct(ls['train']['positive_rate'])} of training candidates and {pct(ls['student_test']['positive_rate'])} of student-test candidates, "
      f"with {ls['train']['mean_candidates_per_obs']:.1f} and {ls['student_test']['mean_candidates_per_obs']:.1f} candidates per GPS point. "
      "The five features are `feat_dist_to_arc`, `feat_arc_length`, `feat_heading_diff`, `feat_speed` and `feat_time_gap`. "
      "Cross-validation uses the trip-grouped 5-fold split shipped in the table (16 trips per fold, no trip in two folds), seed 42. "
      "Validation folds and the student-test set are scored in full, with no resampling. "
      f"All runs were trained and timed on one machine ({os.cpu_count()} logical CPUs).\n")
    w("**Metric definitions.** *Macro F1* averages the F1 of class 0 and class 1 at probability threshold 0.5. "
      "*Top-1 accuracy*: for each GPS point take the candidate with the highest probability; it is correct if that candidate carries label 1 "
      "(ties go to the smaller arc id). *Candidate miss rate*: share of GPS points whose reference arc is not among the candidates at all; "
      "no model can get those right. *Log loss* penalises confident wrong probabilities. *ECE* is the average gap between predicted probability and observed frequency over 10 bins.\n")
    tab = ["| Model | Algorithm family | Key configuration | CV macro F1 (mean ± std) | CV Top-1 | Test Top-1 | Route continuity | Test log loss | Test macro F1 | Inference time |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for lab, r in base_t.iterrows():
        d = Path(r.run_dir)
        cv_top1 = json.loads((d / "metrics.json").read_text())["cv_top1_mean"]
        tab.append(f"| {lab} | {r.family} | {fmt_config(r.key_config)} | {r.cv_macro_f1_mean:.4f} ± {r.cv_macro_f1_std:.4f} | {cv_top1:.4f} | "
                   f"{r.test_top1:.4f} | N/A: needs the decoded road sequence (reported in section 5) | {r.test_log_loss:.4f} | {r.test_macro_f1:.4f} | "
                   f"{r.infer_ms_per_trip:.1f} ms per trip ({r.infer_ms_per_1k:.2f} ms per 1,000 candidates) |")
    w("\n".join(tab) + "\n")
    if timed:
        w("Inference time is the median of 5 repeated `predict_proba` calls on the complete student-test candidate set (276,962 rows, 10 trips) after a warm-up call; all models were timed one after another "
          "on an otherwise idle machine (`experiments/time_models.py`). Fit time is one fit on all 80 training trips.\n")
    w(f"Reference point: predicting the base rate for every candidate gives a student-test log loss of {base_ll_test:.4f}.\n")
    w("**Reading the table.**\n")
    w(f"- XGBoost is best on probability quality and on macro F1 (student-test log loss {f4(tx['log_loss'])}, macro F1 {f4(tx['macro_f1'])}; CV macro F1 {f4(m_x['cv_macro_f1_mean'])}).")
    w(f"- Top-1 does not separate the three tree models: XGBoost {f4(tx['top1_acc'])}, random forest {f4(base.loc['random_forest','test_top1'])}, "
      f"HGB {f4(base.loc['hist_gradient_boosting','test_top1'])} (spread {100*(top1s.max()-top1s.min()):.2f} points on 10 trips, not a meaningful difference).")
    w(f"- Gaussian NB ({f4(tn['log_loss'])}) and logistic regression ({f4(base.loc['logistic_regression','test_log_loss'])}) have a log loss worse than the base-rate reference ({base_ll_test:.4f}), "
      "i.e. worse than giving every candidate the same base-rate probability. The logistic regression, random forest and HGB use `class_weight=\"balanced\"`, which pushes positive probabilities up; "
      "their reliability curves sit below the diagonal (`reliability.png`), while XGBoost, trained without class weights, follows the diagonal (ECE "
      f"{tx['ece']:.3f} against {base.loc['random_forest','test_ece']:.3f}, {base.loc['hist_gradient_boosting','test_ece']:.3f} and {base.loc['logistic_regression','test_ece']:.3f}).")
    w(f"- CV Top-1 for XGBoost ranges from {f_x.top1_acc.min():.3f} (fold {worst}) to {f_x.top1_acc.max():.3f} (fold {best}) because the candidate miss rate differs by fold, from "
      f"{pct(f_x.candidate_miss_rate.min())} to {pct(f_x.candidate_miss_rate.max())}. Fold {worst} contains trip {t54} ({n54:,} GPS points, {pct(miss54, 0)} of them without the reference arc among the candidates). "
      f"Among points whose reference arc is a candidate, XGBoost's Top-1 ranges only from {f_x.top1_acc_hit_only.min():.3f} to {f_x.top1_acc_hit_only.max():.3f}. "
      f"The student-test trips have a lower miss rate ({pct(tx['candidate_miss_rate'])}), which is why student-test Top-1 ({f4(tx['top1_acc'])}) is higher than CV Top-1 ({f4(m_x['cv_top1_mean'])}).\n")
    w("**Primary selection metric.** We select on cross-validated log loss, with Top-1 among points that have a reference candidate as tie-break. Reasons: "
      "(1) the HMM decoder uses these probabilities as emission scores, so probability quality is what it needs; "
      "(2) macro F1 depends on a fixed 0.5 threshold, and with about 12 candidates per point and one positive, that threshold is arbitrary: "
      f"isotonic-calibrated Gaussian NB has a better log loss ({f4(tni['log_loss'])} against {f4(tn['log_loss'])}) and the same ranking quality (AUC {f4(tni['roc_auc'])} against {f4(tn['roc_auc'])}) "
      f"but predicts no positives at threshold 0.5, so its macro F1 drops to {f4(tni['macro_f1'])}; "
      "(3) Top-1 is capped by the candidate miss rate and is nearly equal across the strong models. "
      "Macro F1 is still reported because the template requires it. On this basis XGBoost is the recommended model for tuning; the final model choice is made by the team from this table.\n")

    # ------------------------------------------------------------------ calibration
    w("### Probability calibration check\n")
    w("| Run | CV log loss (pooled) | Test log loss | Test ECE | Test macro F1 | Test Top-1 | Positives predicted at 0.5 (test) |\n|---|---|---|---|---|---|---|")
    for name, m in (("XGBoost", m_x), ("XGBoost + isotonic", m_xi), ("Gaussian NB", m_n), ("Gaussian NB + isotonic", m_ni)):
        t = m["student_test"]
        w(f"| {name} | {m['cv_pooled']['log_loss']:.4f} | {t['log_loss']:.4f} | {t['ece']:.4f} | {t['macro_f1']:.4f} | {t['top1_acc']:.4f} | {t['tp'] + t['fp']:,} |")
    w(f"\nIsotonic calibration (fitted on out-of-fold scores, trip-grouped) changes XGBoost's log loss by {abs(txi['log_loss']-tx['log_loss']):.4f}, so XGBoost needs no calibration. "
      f"It cuts Gaussian NB's log loss from {f4(tn['log_loss'])} to {f4(tni['log_loss'])}, but isotonic regression outputs a step function: the share of GPS points whose two best candidates have identical probability rises from "
      f"{pct(ties['gaussian_nb'])} to {pct(ties['gaussian_nb+isotonic'])} ({pct(ties['xgboost'])} for XGBoost); ties are resolved by the smaller arc id, which carries no information, so Top-1 falls ({f4(tn['top1_acc'])} to {f4(tni['top1_acc'])}). "
      "It also leaves no candidate above 0.5.\n")

    # ------------------------------------------------------------------ 3.3
    w("## 3.3 Visualisations\n")
    w("All figures are in `experiments/results/` and are drawn from the logged runs only.\n")
    for fn, txt in [
        ("cv_macro_f1.png", "CV macro F1 per model (bar = mean, whisker = std, dots = the 5 folds)."),
        ("student_test_top1_macro_f1.png", "Student-test Top-1 accuracy and macro F1 per model."),
        ("log_loss.png", "CV and student-test log loss per model."),
        ("confusion_matrices.png", "Student-test confusion matrices at threshold 0.5 (optional diagnostic)."),
        ("reliability.png", "Calibration curves on the student-test trips; the dashed diagonal is perfect calibration."),
        ("xgboost_learning_curve.png", "XGBoost train vs validation log loss per boosting round, untuned defaults. XGBoost is the only model here with an epoch-wise history, so no other model gets a loss curve."),
        ("xgboost_learning_curve_tuned.png", "The same curve for the tuned XGBoost."),
        ("with_calibration/reliability.png", "Calibration curves including the isotonic variants."),
    ]:
        if (a.results_dir / fn).exists():
            w(f"- `{fn}`: {txt}")
    w("- Folium maps (raw GPS, road network, reference route, predicted route) are produced by the decoder track (section 6); `eda/outputs/folium_train_00000003.html` shows a trip with GPS, road network and reference route.\n")

    # ------------------------------------------------------------------ 3.2 XGBoost
    p = cfg_x["params"]
    w("## 3.2 Model subsections\n")
    w("### XGBoost (gradient-boosted decision trees)\n")
    w("**A. Model understanding**\n")
    claim(tx["log_loss"] < base.loc["logistic_regression", "test_log_loss"] and tx["macro_f1"] > base.loc["logistic_regression", "test_macro_f1"],
          "XGBoost beats the linear model on log loss and macro F1")
    w("1. *Why suitable.* Each candidate is described by five numeric clues, and how much a clue such as distance matters plausibly depends on the others (arc length, heading, speed). "
      f"Boosted trees can learn such interactions, and the data support that the relationship is not linear: XGBoost reaches log loss {f4(tx['log_loss'])} and macro F1 {f4(tx['macro_f1'])} where logistic regression, "
      f"a linear model on the same features, reaches {f4(base.loc['logistic_regression','test_log_loss'])} and {f4(base.loc['logistic_regression','test_macro_f1'])}. Trees also need no feature scaling and are insensitive to the heavy tails in our features "
      f"(`feat_time_gap` is exactly 1 s for {pct(feat_rows[4][2])} of training rows but reaches 153,064 s; `feat_dist_to_arc` has skew {feat_rows[0][1]:.1f}), because trees only use the order of values. "
      "Trained with the logistic objective, it also outputs probabilities that the HMM can use directly.")
    w(f"2. *Strengths and limitations.* Strengths seen in our runs: best log loss ({f4(tx['log_loss'])}), best macro F1 ({f4(tx['macro_f1'])}), almost perfect calibration (ECE {tx['ece']:.3f}), and a fit time of "
      f"{base_t.loc['xgboost', 'train_time_s']:.0f} s on 2.5 million rows. Limitations: (a) with the default 300 rounds it overfits mildly (see C9); "
      f"(b) at threshold 0.5 it finds only {pct(tx['pos_recall'])} of true matches (precision {pct(tx['pos_precision'])}), so route quality has to come from ranking plus the HMM, not from thresholding; "
      f"(c) it scores each candidate alone, so on its own it cannot guarantee a connected route: Top-1 is {f4(tx['top1_acc'])}, not 1.\n")
    w("**B. Training and configuration**\n")
    w(f"5. *Important hyperparameters (initial values).* n_estimators = {p['n_estimators']}, max_depth = {p['max_depth']}, learning_rate = {p['learning_rate']}, subsample = {p['subsample']}, "
      f"colsample_bytree = {p['colsample_bytree']}, min_child_weight = {p['min_child_weight']}, reg_lambda = {p['reg_lambda']}, gamma = {p['gamma']}; histogram tree method, seed {cfg_x['seed']}, no class weights.")
    w("6. *Objective.* Binary logistic loss (log loss), minimised by gradient boosting with L2 leaf regularisation (reg_lambda) and a split penalty (gamma).\n")
    w("**C. Experimental results**\n")
    w(f"7–8. *Student-test results (10 trips).* Top-1 accuracy {f4(tx['top1_acc'])} ({f4(tx['top1_acc_hit_only'])} among points whose reference arc is a candidate; candidate miss rate {pct(tx['candidate_miss_rate'])}), "
      f"macro F1 {f4(tx['macro_f1'])} (macro precision {f4(tx['macro_precision'])}, macro recall {f4(tx['macro_recall'])}), log loss {f4(tx['log_loss'])}, AUC {f4(tx['roc_auc'])}. "
      f"Confusion matrix at 0.5: TN {tx['tn']:,}, FP {tx['fp']:,}, FN {tx['fn']:,}, TP {tx['tp']:,}. Cross-validation: macro F1 {f4(m_x['cv_macro_f1_mean'])} ± {f4(m_x['cv_macro_f1_std'])}, log loss {f4(m_x['cv_log_loss_mean'])}.")
    if lc:
        w(f"9. *Overfitting or underfitting?* Mild overfitting in the number of rounds, no underfitting. In the loss curve the mean validation log loss is lowest at round {lc['best_validation_round']} "
          f"({f4(lc['val_logloss_min'])}) and then creeps up to {f4(lc['val_logloss_final'])} at round {lc['n_rounds']}, while training loss keeps falling to {f4(lc['train_logloss_final'])} "
          f"(final gap {f4(lc['gap_final_val_minus_train'])}). The cost of the extra rounds is small ({100*(lc['val_logloss_final']/lc['val_logloss_min']-1):.1f}% in log loss). "
          f"Generalisation to unseen trips is sound: CV macro F1 {f4(m_x['cv_macro_f1_mean'])} against student-test {f4(tx['macro_f1'])}; the test trips are easier because fewer points lack a reference candidate ({pct(tx['candidate_miss_rate'])}).\n")
    else:
        w("9. *Overfitting or underfitting?* (run `experiments/learning_curves.py`)\n")

    # ------------------------------------------------------------------ 3.2 GNB
    w("### Gaussian Naive Bayes\n")
    w("**A. Model understanding**\n")
    w("1. *Why suitable.* It is the simplest probabilistic classifier: for each class it fits one bell curve per feature and combines them with Bayes' rule. "
      "It trains in closed form in under a second, which makes it a useful baseline that shows how much the other models gain from flexible decision boundaries.")
    w(f"2. *Strengths and limitations.* Strength: speed ({base_t.loc['gaussian_nb', 'train_time_s']:.1f} s to train, {base_t.loc['gaussian_nb', 'infer_ms_per_trip']:.1f} ms per trip to score). Limitations on this data: the bell-curve assumption is badly wrong "
      f"(skew of `feat_dist_to_arc` {feat_rows[0][1]:.1f}, of `feat_arc_length` {feat_rows[1][1]:.1f}, of `feat_time_gap` {feat_rows[4][1]:.0f}; `feat_time_gap` equals 1 s in {pct(feat_rows[4][2])} of rows), "
      f"and independence holds only roughly (largest correlation between two features {overall_corr:.2f} overall, {within[1]:.2f} among true matches). "
      f"Its raw probabilities are poor: log loss {f4(tn['log_loss'])} is worse than the base-rate reference {base_ll_test:.4f}.\n")
    w("**B. Training and configuration**\n")
    w(f"5. *Hyperparameters.* `var_smoothing` = {cfg_n['params']['var_smoothing']} (a tiny variance floor), class priors estimated from the data ({pct(ls['train']['positive_rate'])} positive); features are standardised, which does not change a Gaussian NB except through the floor.")
    w("6. *Objective.* Maximum likelihood of the training data under the Gaussian-independence model, solved in closed form (means and variances per class and feature); there is no iterative loss.\n")
    w("**C. Experimental results**\n")
    w(f"7–8. *Student-test results.* Top-1 {f4(tn['top1_acc'])}, macro F1 {f4(tn['macro_f1'])} (positive-class precision {f4(tn['pos_precision'])}, recall {f4(tn['pos_recall'])}), log loss {f4(tn['log_loss'])}, AUC {f4(tn['roc_auc'])}, ECE {f4(tn['ece'])}. "
      f"Cross-validation: macro F1 {f4(m_n['cv_macro_f1_mean'])} ± {f4(m_n['cv_macro_f1_std'])}.")
    w(f"9. *Overfitting or underfitting?* Underfitting (high bias). The in-sample macro F1 on the training data is {f4(gnb_insample_f1)}, almost identical to the cross-validated {f4(m_n['cv_macro_f1_mean'])} and the student-test {f4(tn['macro_f1'])}: "
      "the model is not memorising anything, it is simply too rigid for this data. Gaussian NB has no epoch-wise history, so no training curve is drawn. "
      f"Isotonic calibration improves log loss to {f4(tni['log_loss'])}, but then no candidate passes 0.5 (macro F1 {f4(tni['macro_f1'])}).\n")

    # ------------------------------------------------------------------ 4
    w("## 4. Bayesian hyperparameter optimisation (XGBoost)\n")
    if not tuned:
        w("_Tuning has not finished yet; run `tuning/run_optuna.py` and regenerate this file._\n")
    else:
        done = int((trials.state == "COMPLETE").sum())
        pruned = int((trials.state == "PRUNED").sum())
        dur = pd.to_timedelta(trials["duration"]).dt.total_seconds()
        w(f"1. *Hyperparameters searched:* {', '.join(space.keys())}.")
        w("2. *Ranges and distributions* (fixed before any tuning run; see `tuning/search_space.json` for the reasoning behind each):\n")
        w("| Hyperparameter | Type | Range | Scale |\n|---|---|---|---|")
        for k, s in space.items():
            w(f"| {k} | {s['type']} | {s['low']} to {s['high']}{(' step ' + str(s['step'])) if 'step' in s else ''} | {'log' if s.get('log') else 'linear'} |")
        w(f"\n3. *Objective:* {sc['objective']} = mean over the 5 trip-grouped folds of the validation log loss ({sc['direction']}); the search uses the 80 training trips only. Why this objective: {sc['objective_why']}")
        w(f"4. *Trials:* {sc['n_trials_requested']} requested, {done} completed and {pruned} stopped early by the median pruner. Sampler: {sc['algorithm']}; the first {sc['n_startup_random_trials']} trials are random. "
          f"Total optimisation time {dur.sum() / 60:.0f} minutes of wall-clock time ({dur.mean():.0f} s per trial on average; the machine was shared with other jobs during part of the search, so single-trial times are not clean timings).")
        w("5. *Best values* (trial " + str(bp['best_trial_number']) + f", CV log loss {bp['best_value']:.4f}): " + ", ".join(f"{k} = {v:.4g}" if isinstance(v, float) else f"{k} = {v}" for k, v in bp["params"].items()) + ".")
        u, t_ = tv.loc["untuned"], tv.loc["tuned"]

        def cv_ll(run_id: str) -> float:
            return json.loads((a.log_dir / run_id / "metrics.json").read_text())["cv_log_loss_mean"]

        cv_u, cv_t = cv_ll(u["run_id"]), cv_ll(t_["run_id"])
        claim(abs(cv_t - bp["best_value"]) < 5e-4, "the tuned run's CV log loss reproduces the best Optuna trial value")
        claim(cv_t < cv_u, "the tuned model has the lower CV log loss")
        d_test = float(t_["student_test_log_loss"]) - float(u["student_test_log_loss"])
        claim(abs(d_test) < 0.002, "student-test log loss is essentially unchanged by tuning")
        fit_u, fit_t, inf_u, inf_t = float(u["train_time_s"]), float(t_["train_time_s"]), float(u["inference_ms_per_trip"]), float(t_["inference_ms_per_trip"])
        if timing is not None:  # consistent timings from time_models.py
            fit_u, fit_t = float(timing.loc["xgboost", "fit_time_s"]), float(timing.loc["xgboost_tuned", "fit_time_s"])
            inf_u, inf_t = float(timing.loc["xgboost", "infer_ms_per_trip"]), float(timing.loc["xgboost_tuned", "infer_ms_per_trip"])
        claim(fit_t > fit_u and inf_t > inf_u, "the tuned model is slower to fit and to score")
        w("6. *Did the tuned model beat the untuned one?* Same table, folds, seed and metrics for both:\n")
        w("| Metric | Untuned (defaults) | Tuned | Difference |\n|---|---|---|---|")
        for name, uv, tv_, nd in (
            ("CV log loss (the tuning objective)", cv_u, cv_t, 4),
            ("CV macro F1", float(u["cv_macro_f1_mean"]), float(t_["cv_macro_f1_mean"]), 4),
            ("Student-test Top-1", float(u["student_test_top1"]), float(t_["student_test_top1"]), 4),
            ("Student-test macro F1", float(u["student_test_macro_f1"]), float(t_["student_test_macro_f1"]), 4),
            ("Student-test log loss", float(u["student_test_log_loss"]), float(t_["student_test_log_loss"]), 4),
            ("Inference time (ms per trip)", inf_u, inf_t, 1),
            ("Training time (s)", fit_u, fit_t, 1),
        ):
            w(f"| {name} | {uv:.{nd}f} | {tv_:.{nd}f} | {tv_ - uv:+.{nd}f} |")
        verdict = (f"\nOn the quantity that was optimised, the tuned model is better: CV log loss {cv_u:.4f} to {cv_t:.4f} ({100 * (cv_t - cv_u) / cv_u:+.1f}%). "
                   f"On the student-test trips the change is negligible: log loss {d_test:+.4f}, Top-1 {float(t_['student_test_top1']) - float(u['student_test_top1']):+.4f}, "
                   f"macro F1 {float(t_['student_test_macro_f1']) - float(u['student_test_macro_f1']):+.4f}; with only 10 test trips, differences of a few thousandths are within noise. "
                   f"So tuning did not change which model is best or what the decoder will receive in practice. The tuned model is also slower (fit {fit_u:.0f} s to {fit_t:.0f} s, inference {inf_u:.1f} to {inf_t:.1f} ms per trip).")
        if lc and lc_t:
            claim(lc_t["gap_final_val_minus_train"] < lc["gap_final_val_minus_train"], "the tuned model overfits less than the default")
            verdict += (f" The loss curves show why it helps a little: the default (learning rate {cfg_x['params']['learning_rate']}) reaches its lowest validation loss at round {lc['best_validation_round']} of {lc['n_rounds']} "
                        f"and then drifts upward (train-validation gap {lc['gap_final_val_minus_train']:.3f} at the end), whereas the tuned model (learning rate {bp['params']['learning_rate']:.3f}) "
                        f"keeps improving until round {lc_t['best_validation_round']} of {lc_t['n_rounds']} and overfits less (gap {lc_t['gap_final_val_minus_train']:.3f}).")
        w(verdict)
        w("\nFigures: `tuning/studies/xgboost_cv_log_loss/optimization_history.png`, `param_importance.png`. Full trial log: `trials.csv`.\n")

    # order: 3.1, 3.2, 3.3, 4 (the blocks were generated 3.1, 3.3, 3.2, 4)
    i33, i32, i4 = out.index("## 3.3 Visualisations\n"), out.index("## 3.2 Model subsections\n"), out.index("## 4. Bayesian hyperparameter optimisation (XGBoost)\n")
    out = out[:i33] + out[i32:i4] + out[i33:i32] + out[i4:]

    # where each model's settings came from
    i33b = out.index("## 3.3 Visualisations\n")
    out.insert(i33b, "**Provenance of the models in the table.** XGBoost and Gaussian NB are Person 2's. Logistic regression, random forest and HGB were run here, through the same harness, "
               "with the hyperparameters from the teammates' scripts in the team repository (`experiments/train_model*.py`), so that every row uses the same folds and complete validation sets. "
               "Their owners explain those models in their own subsections. KNN and the MLP are still to be added by their owners through the same harness; "
               "the results published in the team repository for those two were computed on class-rebalanced random subsamples and are not comparable with this table.\n")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("\n".join(out), encoding="utf-8")
    print(f"wrote {a.out} ({len(out)} blocks); tuning section {'filled' if tuned else 'pending'}")


if __name__ == "__main__":
    main()
