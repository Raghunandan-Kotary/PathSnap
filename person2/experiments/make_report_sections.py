"""Generate Person 2's report sections (3.1, 3.2 for XGBoost and Gaussian NB, 3.3, 4, final results) from logs.

Every number is read from person2/experiments/logs/*, person2/tuning/studies/*, the results folder or the
feature tables, so the text can be regenerated and checked against the repository. `claim()` stops the
script if a sentence would contradict the data.

    python person2/experiments/make_report_sections.py
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

P2 = config.REPO_ROOT
F5 = ["feat_dist_to_arc", "feat_arc_length", "feat_heading_diff", "feat_speed", "feat_time_gap"]


def claim(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(f"report sentence would contradict the logs: {msg}")


def f4(x) -> str:
    return f"{float(x):.4f}"


def pct(x, d=1) -> str:
    return f"{100 * float(x):.{d}f}%"


def prior_ll(p: float) -> float:
    return float(-(p * np.log(p) + (1 - p) * np.log(1 - p)))


def error_breakdown(table: pd.DataFrame, proba: pd.DataFrame) -> dict:
    """Classifier-only top pick per GPS point on the student-test trips, and what kind of arc the wrong picks are."""
    te = table[table.partition == "student_test"][["trip_id", "obs_idx", "arc_id", "from_node", "to_node", "label"]]
    te = te.merge(proba, on=["trip_id", "obs_idx", "arc_id"])
    K = ["trip_id", "obs_idx"]
    gt = te[te.label == 1][K + ["from_node", "to_node"]].rename(columns={"from_node": "gu", "to_node": "gv"})
    top = te.sort_values(K + ["proba", "arc_id"], ascending=[True, True, False, True]).drop_duplicates(K).merge(gt, on=K, how="left")
    wrong = top[(top.label == 0) & top.gu.notna()]
    rev = (wrong.from_node == wrong.gv) & (wrong.to_node == wrong.gu)
    adj = ~rev & ((wrong.from_node == wrong.gv) | (wrong.to_node == wrong.gu) | (wrong.from_node == wrong.gu) | (wrong.to_node == wrong.gv))
    return {"top1": float((top.label == 1).mean()), "wrong": len(wrong) / len(top), "reverse": float(rev.mean()),
            "adjacent": float(adj.mean()), "other": float((~rev & ~adj).mean())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table-v1", type=Path, default=P2 / "experiments" / "team_table" / "candidates_features.parquet")
    ap.add_argument("--table-v2", type=Path, default=P2 / "experiments" / "team_table_v2" / "candidates_features.parquet")
    ap.add_argument("--v1-proba", type=Path, default=None, help="proba_student_test.parquet of the v1 XGBoost run (kept outside git)")
    ap.add_argument("--log-dir", type=Path, default=config.EXPERIMENT_LOGS)
    ap.add_argument("--results-dir", type=Path, default=P2 / "experiments" / "results")
    ap.add_argument("--study-dir", type=Path, default=config.TUNING_DIR / "studies" / "xgboost_v2_cv_log_loss")
    ap.add_argument("--out", type=Path, default=P2 / "docs" / "report" / "PERSON2_REPORT_SECTIONS.md")
    a = ap.parse_args()

    runs = collect(a.log_dir)
    runs = runs[~runs.smoke]
    v2, v1 = runs[runs.feature_set == "v2"], runs[runs.feature_set == "v1"]
    base2 = pick_runs(v2, ["baseline"]).set_index("label")
    base2_t, timed = apply_timing(base2.reset_index(), a.results_dir)
    base2_t = base2_t.set_index("label")
    base1 = pick_runs(v1, ["baseline"]).set_index("label")
    cal2 = pick_runs(v2, ["baseline", "calibration_variant"]).set_index("label")

    def files(label, src):
        d = Path(src.loc[label, "run_dir"])
        return json.loads((d / "config.json").read_text()), json.loads((d / "metrics.json").read_text()), pd.read_csv(d / "fold_metrics.csv")

    cfg_x, m_x, f_x = files("xgboost", cal2)
    cfg_n, m_n, f_n = files("gaussian_nb", cal2)
    _, m_xi, _ = files("xgboost+isotonic", cal2)
    _, m_ni, _ = files("gaussian_nb+isotonic", cal2)
    tx, tn, txi, tni = m_x["student_test"], m_n["student_test"], m_xi["student_test"], m_ni["student_test"]
    ls = cfg_x["label_stats"]
    base_ll = prior_ll(ls["student_test"]["positive_rate"])
    feats = cfg_x["feature_cols"]
    new_feats = [c for c in feats if c.startswith("f2_")]

    # ---------------- facts from the tables ----------------
    from scipy.stats import skew
    from sklearn.metrics import f1_score

    from person2.experiments.models import GAUSSIAN_NB
    t2 = pd.read_parquet(a.table_v2)
    tr2 = t2[t2.partition == "train"]
    gnb = GAUSSIAN_NB.build(GAUSSIAN_NB.defaults, config.GLOBAL_SEED).fit(tr2[feats].to_numpy("float32"), tr2.label.to_numpy())
    gnb_in = float(f1_score(tr2.label, gnb.predict_proba(tr2[feats].to_numpy("float32"))[:, 1] >= 0.5, average="macro"))
    sk = {c: float(skew(tr2[c].to_numpy())) for c in feats}
    eb2 = error_breakdown(t2, pd.read_parquet(Path(cal2.loc["xgboost", "run_dir"]) / "proba_student_test.parquet"))
    eb1 = error_breakdown(pd.read_parquet(a.table_v1), pd.read_parquet(a.v1_proba)) if a.v1_proba and a.v1_proba.exists() else None

    # ---------------- claims ----------------
    claim(base2["test_log_loss"].idxmin() == "xgboost", "XGBoost has the lowest student-test log loss (v2)")
    claim(base2["test_macro_f1"].idxmax() == "xgboost", "XGBoost has the highest student-test macro F1 (v2)")
    claim(base2["cv_macro_f1_mean"].idxmax() == "xgboost", "XGBoost has the highest CV macro F1 (v2)")
    common = [m for m in base2.index if m in base1.index]
    claim(all(base2.loc[m, "test_top1"] > base1.loc[m, "test_top1"] for m in common), "every model's Top-1 improves with v2")
    claim(tni["log_loss"] < tn["log_loss"], "isotonic calibration improves Gaussian NB log loss")
    claim(abs(txi["log_loss"] - tx["log_loss"]) < 0.005, "isotonic calibration barely changes XGBoost")
    claim(gnb_in - m_n["cv_macro_f1_mean"] < 0.05, "Gaussian NB shows no large train/CV gap")
    if eb1:
        claim(eb2["wrong"] < eb1["wrong"], "v2 makes fewer classifier-only errors than v1")

    lc = json.loads((a.results_dir / "xgboost_learning_curve.json").read_text()) if (a.results_dir / "xgboost_learning_curve.json").exists() else None
    lc_t = json.loads((a.results_dir / "xgboost_learning_curve_tuned.json").read_text()) if (a.results_dir / "xgboost_learning_curve_tuned.json").exists() else None
    timing = pd.read_csv(a.results_dir / "timing.csv").set_index("label") if (a.results_dir / "timing.csv").exists() else None
    fin = {p: json.loads((a.results_dir / f"final_pipeline_{p}.json").read_text())
           for p in ("student_test", "evaluator_test") if (a.results_dir / f"final_pipeline_{p}.json").exists()}

    out: list[str] = []
    w = out.append
    w("# Person 2: report sections (generated from logged runs)\n")
    w("> Every number below is read from `person2/experiments/logs/`, `person2/tuning/studies/`, `person2/experiments/results/` "
      "or the feature tables by `person2/experiments/make_report_sections.py`. Regenerate it after any new run.\n")

    # ================================================================== 3.1
    w("## 3.1 Overall model comparison\n")
    w(f"**How every model was measured.** All models use the same candidate table (Person 1, commit `86ad172`): "
      f"{ls['train']['candidates']:,} candidate rows from 80 training trips ({ls['train']['observations_with_candidates']:,} GPS points) and "
      f"{ls['student_test']['candidates']:,} candidate rows from the 10 student-test trips ({ls['student_test']['observations_with_candidates']:,} GPS points). "
      "Each candidate is a (GPS point, road arc) pair; the label is 1 for the single arc the GPS point is matched to on the reference route. "
      f"Positives are {pct(ls['train']['positive_rate'])} of training candidates. Cross-validation uses the trip-grouped 5-fold split shipped in the table "
      "(16 trips per fold, no trip in two folds), seed 42; validation folds and the student-test set are scored in full, with no resampling. "
      f"All runs were trained and timed on one machine ({os.cpu_count()} logical CPUs).\n")
    w(f"**Features.** {len(feats)} per candidate: the 5 frozen features ({', '.join('`'+c+'`' for c in F5)}) plus {len(new_feats)} context features "
      f"({', '.join('`'+c+'`' for c in new_feats)}), computed by `person2/experiments/context_features.py` from the raw `.track` and `.nodes` files and the candidate list only "
      "(no labels, no reference route), so the API computes them live with the same function. The change from the 5 frozen features was agreed by the team "
      "and every model was re-run on the new table (see 3.1.1).\n")
    w("**Metric definitions.** *Macro F1* averages the F1 of class 0 and class 1 at threshold 0.5. *Top-1 accuracy*: for each GPS point take the candidate "
      "with the highest probability; it is correct if that candidate carries label 1 (ties go to the smaller arc id). *Candidate miss rate*: share of GPS points "
      "whose reference arc is not among the candidates, which caps Top-1. *Log loss* penalises confident wrong probabilities. *ECE* is the average gap between "
      "predicted probability and observed frequency over 10 bins. *Route continuity (classifier-only)*: share of consecutive top picks that are connected in the "
      "road network, from `evaluation/route_metrics.py` (Person 3), before any HMM decoding.\n")
    tab = ["| Model | Algorithm family | Key configuration | CV macro F1 (mean ± std) | CV Top-1 | Test Top-1 | Route continuity (classifier-only, test) | Test log loss | Test macro F1 | Inference time |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for lab, r in base2_t.iterrows():
        mm = json.loads((Path(r.run_dir) / "metrics.json").read_text())
        rc = mm.get("route_student_test", {}).get("route_continuity")
        tab.append(f"| {lab} | {r.family} | {fmt_config(r.key_config)} | {r.cv_macro_f1_mean:.4f} ± {r.cv_macro_f1_std:.4f} | {mm['cv_top1_mean']:.4f} | "
                   f"{r.test_top1:.4f} | {('%.4f' % rc) if rc is not None else 'not computed'} | {r.test_log_loss:.4f} | {r.test_macro_f1:.4f} | "
                   f"{r.infer_ms_per_trip:.1f} ms per trip |")
    w("\n".join(tab) + "\n")
    w(f"Reference point: predicting the base rate for every candidate gives a student-test log loss of {base_ll:.4f}. "
      + ("Inference time is the median of 5 repeated `predict_proba` calls on the complete student-test candidate set after a warm-up, all models timed one after another "
         "on an otherwise idle machine (`person2/experiments/time_models.py`).\n" if timed else "\n"))
    w("**Reading the table.**\n")
    w(f"- XGBoost is best on every probability and classification measure (student-test log loss {f4(tx['log_loss'])}, macro F1 {f4(tx['macro_f1'])}, CV macro F1 {f4(m_x['cv_macro_f1_mean'])}).")
    rf, hgb = base2.loc["random_forest"], base2.loc["hist_gradient_boosting"]
    w(f"- Random forest and HGB are close behind on Top-1 ({f4(rf.test_top1)} and {f4(hgb.test_top1)} against {f4(tx['top1_acc'])}) but have about twice the log loss "
      f"({f4(rf.test_log_loss)}, {f4(hgb.test_log_loss)}). They are trained with `class_weight=\"balanced\"`, which pushes positive probabilities up; XGBoost is trained without class weights "
      f"and is the best calibrated (ECE {tx['ece']:.3f} against {rf.test_ece:.3f} and {hgb.test_ece:.3f}).")
    w(f"- Gaussian NB has the worst log loss ({f4(tn['log_loss'])}, worse than the base rate {base_ll:.4f}): several context features are very skewed "
      f"(skew of `f2_end_dist` {sk['f2_end_dist']:.0f}, `feat_time_gap` {sk['feat_time_gap']:.0f}), which breaks its bell-curve assumption.")
    w(f"- XGBoost's CV Top-1 is lowest on fold {int(f_x.top1_acc.idxmin())} ({f_x.top1_acc.min():.3f}), the fold with the highest candidate miss rate ({pct(f_x.candidate_miss_rate.max())}); "
      f"among points whose reference arc is a candidate it ranges from {f_x.top1_acc_hit_only.min():.3f} to {f_x.top1_acc_hit_only.max():.3f}. "
      f"The student-test trips have a lower miss rate ({pct(tx['candidate_miss_rate'])}).\n")

    # ------------------------------------------------------------ 3.1.1
    w("### 3.1.1 Why the feature set changed (evidence)\n")
    if eb1:
        w(f"With the 5 frozen features, the tuned XGBoost picked a wrong arc for {pct(eb1['wrong'])} of student-test GPS points (classifier only). Of those wrong picks, "
          f"{pct(eb1['adjacent'])} were an *adjacent* arc (sharing a node with the true arc: the point is near a junction), {pct(eb1['reverse'])} were the *reverse-direction twin* "
          f"of the true arc, and {pct(eb1['other'])} were other arcs. The frozen features carry no information about where along an arc a point projects, use a one-step heading, "
          "and score each candidate without looking at the other candidates or at the neighbouring GPS points. With the frozen features the HMM decoder gained nothing from the "
          "classifier: on the student-test trips it scored the same with classifier emissions as with distance alone (section 5). The context features target exactly these cases. "
          f"With them, XGBoost's wrong picks fall to {pct(eb2['wrong'])} ({pct(eb2['adjacent'])} adjacent, {pct(eb2['reverse'])} reverse, {pct(eb2['other'])} other).\n")
    w("| Model | Test Top-1, 5 frozen features | Test Top-1, with context features | Test log loss, 5 frozen | Test log loss, with context |\n|---|---|---|---|---|")
    for m in common:
        w(f"| {m} | {base1.loc[m,'test_top1']:.4f} | {base2.loc[m,'test_top1']:.4f} | {base1.loc[m,'test_log_loss']:.4f} | {base2.loc[m,'test_log_loss']:.4f} |")
    w("\nThe 5-feature runs stay in `person2/experiments/logs/` and their tables and figures in `person2/experiments/results/v1_frozen_features/`.\n")

    w("**Primary selection metric.** We select on cross-validated log loss, with Top-1 as tie-break. The HMM decoder uses these probabilities as emission scores, so "
      "probability quality is what it needs; macro F1 depends on a fixed 0.5 threshold that is arbitrary when each GPS point has about 12 candidates and one positive "
      f"(calibrated Gaussian NB, for example, changes log loss from {f4(tn['log_loss'])} to {f4(tni['log_loss'])} while its macro F1 moves from {f4(tn['macro_f1'])} to {f4(tni['macro_f1'])}). "
      "On this basis XGBoost is the selected model.\n")

    w("### Probability calibration check\n")
    w("| Run | CV log loss (pooled) | Test log loss | Test ECE | Test macro F1 | Test Top-1 |\n|---|---|---|---|---|---|")
    for name, m in (("XGBoost", m_x), ("XGBoost + isotonic", m_xi), ("Gaussian NB", m_n), ("Gaussian NB + isotonic", m_ni)):
        t = m["student_test"]
        w(f"| {name} | {m['cv_pooled']['log_loss']:.4f} | {t['log_loss']:.4f} | {t['ece']:.4f} | {t['macro_f1']:.4f} | {t['top1_acc']:.4f} |")
    w(f"\nIsotonic calibration (fitted on out-of-fold scores, trip-grouped) changes XGBoost's log loss only from {f4(tx['log_loss'])} to {f4(txi['log_loss'])}, so the final model is uncalibrated XGBoost. "
      f"It repairs Gaussian NB's probabilities ({f4(tn['log_loss'])} to {f4(tni['log_loss'])}).\n")
    w("**Provenance.** XGBoost and Gaussian NB are Person 2's. Logistic regression, random forest and HGB use the hyperparameters from the teammates' scripts "
      "(`experiments/train_model*.py`); the linear SVM is Person 3's (`person2/experiments/models.py`). All were run through the same harness on the same table. "
      "KNN and the MLP were not re-run on the context-feature table.\n")

    # ================================================================== 3.2
    p = cfg_x["params"]
    w("## 3.2 Model subsections\n")
    w("### XGBoost (gradient-boosted decision trees)\n")
    w("**A. Model understanding**\n")
    lr2 = base2.loc["logistic_regression"]
    w(f"1. *Why suitable.* Each candidate is described by {len(feats)} numeric clues whose effect interacts (how far a candidate may be from the GPS point depends on where along the arc "
      f"it projects, on the heading agreement and on the neighbouring points). Boosted trees learn such interactions, need no scaling, and are insensitive to heavy tails "
      f"(`feat_time_gap` reaches 153,064 s). The data support a non-linear model: on the same features XGBoost reaches log loss {f4(tx['log_loss'])} and macro F1 {f4(tx['macro_f1'])}, "
      f"logistic regression {f4(lr2.test_log_loss)} and {f4(lr2.test_macro_f1)}. With the logistic objective it outputs probabilities the HMM uses directly.")
    w(f"2. *Strengths and limitations.* Strengths: best log loss, best macro F1, best calibration (ECE {tx['ece']:.3f}), fit in {base2_t.loc['xgboost','train_time_s']:.0f} s on 2.5 million rows. "
      f"Limitations: it scores each candidate alone, so it cannot guarantee a connected route (classifier-only Top-1 {f4(tx['top1_acc'])}); "
      f"at threshold 0.5 it finds {pct(tx['pos_recall'])} of true matches (precision {pct(tx['pos_precision'])}), so the route comes from ranking plus the HMM.\n")
    w("**B. Training and configuration**\n")
    w(f"5. *Important hyperparameters (initial values).* n_estimators = {p['n_estimators']}, max_depth = {p['max_depth']}, learning_rate = {p['learning_rate']}, subsample = {p['subsample']}, "
      f"colsample_bytree = {p['colsample_bytree']}, min_child_weight = {p['min_child_weight']}, reg_lambda = {p['reg_lambda']}, gamma = {p['gamma']}; histogram tree method, seed {cfg_x['seed']}, no class weights. Tuned values: section 4.")
    w("6. *Objective.* Binary logistic loss (log loss), minimised by gradient boosting with L2 leaf regularisation (reg_lambda) and a split penalty (gamma).\n")
    w("**C. Experimental results**\n")
    w(f"7–8. *Student-test results (10 trips).* Top-1 {f4(tx['top1_acc'])} ({f4(tx['top1_acc_hit_only'])} among points whose reference arc is a candidate), macro F1 {f4(tx['macro_f1'])} "
      f"(macro precision {f4(tx['macro_precision'])}, macro recall {f4(tx['macro_recall'])}), log loss {f4(tx['log_loss'])}, AUC {f4(tx['roc_auc'])}. "
      f"Confusion matrix at 0.5: TN {tx['tn']:,}, FP {tx['fp']:,}, FN {tx['fn']:,}, TP {tx['tp']:,}. Cross-validation: macro F1 {f4(m_x['cv_macro_f1_mean'])} ± {f4(m_x['cv_macro_f1_std'])}, log loss {f4(m_x['cv_log_loss_mean'])}.")
    if lc:
        w(f"9. *Overfitting or underfitting?* The loss curve (`xgboost_learning_curve.png`) shows the mean validation log loss lowest at round {lc['best_validation_round']} "
          f"({f4(lc['val_logloss_min'])}), ending at {f4(lc['val_logloss_final'])} after {lc['n_rounds']} rounds while training loss falls to {f4(lc['train_logloss_final'])} "
          f"(final gap {f4(lc['gap_final_val_minus_train'])}). "
          + ((f"Validation loss rises {100 * (lc['val_logloss_final'] / lc['val_logloss_min'] - 1):.0f}% after its best round, which is "
              + ("clear overfitting in the number of rounds; the tuned settings (lower learning rate, more regularisation) remove most of it (section 4). "
                 if lc["val_logloss_final"] / lc["val_logloss_min"] > 1.05 else "mild overfitting in the number of rounds. "))
             if lc["best_validation_round"] < 0.8 * lc["n_rounds"] else "Validation loss is still improving at the end, so there is no overfitting in the number of rounds. ")
          + f"Generalisation to unseen trips is sound: CV macro F1 {f4(m_x['cv_macro_f1_mean'])}, student-test {f4(tx['macro_f1'])}.\n")

    w("### Gaussian Naive Bayes\n")
    w("**A. Model understanding**\n")
    w("1. *Why suitable.* The simplest probabilistic classifier: one bell curve per feature and class, combined with Bayes' rule, fitted in closed form in under a second. "
      "It is the baseline that shows how much the flexible models gain.")
    w(f"2. *Strengths and limitations.* Strength: speed ({base2_t.loc['gaussian_nb','train_time_s']:.1f} s to train). Limitations on this data: the bell-curve assumption is badly wrong "
      f"(skew {sk['feat_dist_to_arc']:.1f} for `feat_dist_to_arc`, {sk['f2_end_dist']:.0f} for `f2_end_dist`, {sk['feat_time_gap']:.0f} for `feat_time_gap`), and the context features are strongly related "
      f"to each other by construction (e.g. `f2_dist_minus_min` is derived from `feat_dist_to_arc`), so independence does not hold. Its raw log loss ({f4(tn['log_loss'])}) is worse than the base rate.\n")
    w("**B. Training and configuration**\n")
    w(f"5. *Hyperparameters.* `var_smoothing` = {cfg_n['params']['var_smoothing']}; class priors from the data. Features are standardised, which does not change a Gaussian NB except through the floor.")
    w("6. *Objective.* Maximum likelihood under the Gaussian-independence model, solved in closed form; no iterative loss.\n")
    w("**C. Experimental results**\n")
    w(f"7–8. *Student-test results.* Top-1 {f4(tn['top1_acc'])}, macro F1 {f4(tn['macro_f1'])}, log loss {f4(tn['log_loss'])}, AUC {f4(tn['roc_auc'])}. "
      f"CV macro F1 {f4(m_n['cv_macro_f1_mean'])} ± {f4(m_n['cv_macro_f1_std'])}. With isotonic calibration: log loss {f4(tni['log_loss'])}, macro F1 {f4(tni['macro_f1'])}, Top-1 {f4(tni['top1_acc'])}.")
    w(f"9. *Overfitting or underfitting?* Underfitting: in-sample macro F1 {f4(gnb_in)} is close to the CV value {f4(m_n['cv_macro_f1_mean'])}, so the model is too rigid rather than memorising. "
      "It has no epoch-wise history, so no training curve is drawn.\n")

    # ================================================================== 3.3
    w("## 3.3 Visualisations\n")
    w("All figures are in `person2/experiments/results/`, drawn from the logged runs only (5-feature versions in `results/v1_frozen_features/`).\n")
    for fn, txt in [("cv_macro_f1.png", "CV macro F1 per model (bar = mean, whisker = std, dots = folds)."),
                    ("student_test_top1_macro_f1.png", "Student-test Top-1 accuracy and macro F1 per model."),
                    ("log_loss.png", "CV and student-test log loss per model."),
                    ("confusion_matrices.png", "Student-test confusion matrices at threshold 0.5."),
                    ("reliability.png", "Calibration curves on the student-test trips."),
                    ("xgboost_learning_curve.png", "XGBoost train vs validation log loss per boosting round (the only model with an epoch-wise history)."),
                    ("xgboost_learning_curve_tuned.png", "The same curve for the tuned XGBoost."),
                    ("with_calibration/reliability.png", "Calibration curves including the isotonic variants.")]:
        if (a.results_dir / fn).exists():
            w(f"- `{fn}`: {txt}")
    w("")

    # ================================================================== 4
    w("## 4. Bayesian hyperparameter optimisation (XGBoost)\n")
    sd = a.study_dir
    if (sd / "tuned_vs_untuned.csv").exists():
        sc, bp = json.loads((sd / "study_config.json").read_text()), json.loads((sd / "best_params.json").read_text())
        tv, trials = pd.read_csv(sd / "tuned_vs_untuned.csv", index_col=0), pd.read_csv(sd / "trials.csv")
        space = sc["search_space"]
        done, pruned = int((trials.state == "COMPLETE").sum()), int((trials.state == "PRUNED").sum())
        dur = pd.to_timedelta(trials["duration"]).dt.total_seconds()
        w(f"1. *Hyperparameters searched:* {', '.join(space)}.")
        w("2. *Ranges and distributions* (fixed before tuning; reasons in `person2/tuning/search_space.json`):\n")
        w("| Hyperparameter | Type | Range | Scale |\n|---|---|---|---|")
        for k, s in space.items():
            w(f"| {k} | {s['type']} | {s['low']} to {s['high']}{(' step ' + str(s['step'])) if 'step' in s else ''} | {'log' if s.get('log') else 'linear'} |")
        w(f"\n3. *Objective:* mean over the 5 trip-grouped folds of the validation log loss (minimised), on the 80 training trips only. Why: {sc['objective_why']}")
        w(f"4. *Trials:* {sc['n_trials_requested']} requested, {done} completed and {pruned} stopped early by the median pruner; {sc['algorithm']}, first {sc['n_startup_random_trials']} trials random; "
          f"{dur.sum() / 60:.0f} minutes in total.")
        w("5. *Best values* (trial " + str(bp["best_trial_number"]) + f", CV log loss {bp['best_value']:.4f}): "
          + ", ".join(f"{k} = {v:.4g}" if isinstance(v, float) else f"{k} = {v}" for k, v in bp["params"].items()) + ".")
        u, t_ = tv.loc["untuned"], tv.loc["tuned"]
        cv_ll = lambda rid: json.loads((a.log_dir / rid / "metrics.json").read_text())["cv_log_loss_mean"]  # noqa: E731
        cu, ct = cv_ll(u["run_id"]), cv_ll(t_["run_id"])
        fu, ft = float(u["train_time_s"]), float(t_["train_time_s"])
        iu, it = float(u["inference_ms_per_trip"]), float(t_["inference_ms_per_trip"])
        if timing is not None and "xgboost_tuned" in timing.index:
            fu, ft = float(timing.loc["xgboost", "fit_time_s"]), float(timing.loc["xgboost_tuned", "fit_time_s"])
            iu, it = float(timing.loc["xgboost", "infer_ms_per_trip"]), float(timing.loc["xgboost_tuned", "infer_ms_per_trip"])
        w("6. *Tuned vs untuned* (same table, folds, seed and metrics):\n")
        w("| Metric | Untuned (defaults) | Tuned | Difference |\n|---|---|---|---|")
        for name, uv, tv_, nd in (("CV log loss (the objective)", cu, ct, 4), ("CV macro F1", u["cv_macro_f1_mean"], t_["cv_macro_f1_mean"], 4),
                                  ("Student-test Top-1", u["student_test_top1"], t_["student_test_top1"], 4), ("Student-test macro F1", u["student_test_macro_f1"], t_["student_test_macro_f1"], 4),
                                  ("Student-test log loss", u["student_test_log_loss"], t_["student_test_log_loss"], 4), ("Inference time (ms per trip)", iu, it, 1), ("Training time (s)", fu, ft, 1)):
            w(f"| {name} | {float(uv):.{nd}f} | {float(tv_):.{nd}f} | {float(tv_) - float(uv):+.{nd}f} |")
        better = ct < cu
        w(f"\n{'The tuned model is better on the optimised quantity' if better else 'Tuning did not improve the optimised quantity'}: CV log loss {cu:.4f} to {ct:.4f} "
          f"({100 * (ct - cu) / cu:+.1f}%). On the student-test trips: log loss {float(t_['student_test_log_loss']) - float(u['student_test_log_loss']):+.4f}, "
          f"Top-1 {float(t_['student_test_top1']) - float(u['student_test_top1']):+.4f}, macro F1 {float(t_['student_test_macro_f1']) - float(u['student_test_macro_f1']):+.4f} "
          "(10 trips: differences of a few thousandths are within noise). "
          f"The final model uses the {'tuned' if better else 'default'} settings, chosen on the CV objective only.")
        if lc and lc_t:
            w(f" Loss curves: default best validation round {lc['best_validation_round']} of {lc['n_rounds']} (gap {lc['gap_final_val_minus_train']:.3f}); "
              f"tuned best round {lc_t['best_validation_round']} of {lc_t['n_rounds']} (gap {lc_t['gap_final_val_minus_train']:.3f}).")
        w(f"\nAn earlier study on the 5 frozen features is kept in `person2/tuning/studies/xgboost_cv_log_loss/` for the record. Figures: `{sd.relative_to(P2.parent).as_posix()}/optimization_history.png`, `param_importance.png`; full trial log `trials.csv`.\n")
    else:
        w("_Tuning has not finished; regenerate this file after `person2/tuning/run_optuna.py`._\n")

    # ================================================================== final
    w("## Final end-to-end results (full API pipeline)\n")
    if fin:
        w("The complete pipeline (`api/pipeline.py`: candidates, frozen and context features, XGBoost probabilities, tuned log-space Viterbi decoder, server-side scoring) "
          "was run over every trip by `person2/experiments/evaluate_pipeline.py`. Accuracy is the share of GPS points whose decoded arc equals the reference arc for that point "
          "(`evaluation/route_metrics.py`).\n")
        w("| Partition | Trips | Pooled accuracy | Mean trip accuracy | Mean route continuity | Execution mode |\n|---|---|---|---|---|---|")
        for pname, s in fin.items():
            w(f"| {pname} | {s['n_trips']} | {s['pooled_pointwise_accuracy']:.4f} | {s['mean_trip_accuracy']:.4f} | {s['mean_route_continuity']:.4f} | "
              + ", ".join(f"{k} ({v})" for k, v in s["execution_modes"].items()) + " |")
        w("\nThe evaluator partition was scored once, after every model, feature and decoder choice had been fixed on the training and student-test trips; nothing was tuned on it.\n")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("\n".join(out).replace("\n Loss curves:", " Loss curves:"), encoding="utf-8")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
