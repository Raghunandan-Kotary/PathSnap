# Person 2: report sections (generated from logged runs)

> Every number below is read from `person2/experiments/logs/`, `person2/tuning/studies/`, `person2/experiments/results/` or the feature tables by `person2/experiments/make_report_sections.py`. Regenerate it after any new run.

## 3.1 Overall model comparison

**How every model was measured.** All models use the same candidate table (Person 1, commit `86ad172`): 2,546,809 candidate rows from 80 training trips (208,417 GPS points) and 276,962 candidate rows from the 10 student-test trips (19,160 GPS points). Each candidate is a (GPS point, road arc) pair; the label is 1 for the single arc the GPS point is matched to on the reference route. Positives are 7.5% of training candidates. Cross-validation uses the trip-grouped 5-fold split shipped in the table (16 trips per fold, no trip in two folds), seed 42; validation folds and the student-test set are scored in full, with no resampling. All runs were trained and timed on one machine (16 logical CPUs).

**Features.** 13 per candidate: the 5 frozen features (`feat_dist_to_arc`, `feat_arc_length`, `feat_heading_diff`, `feat_speed`, `feat_time_gap`) plus 8 context features (`f2_proj_frac`, `f2_end_dist`, `f2_cos_heading`, `f2_ctx_dist`, `f2_dist_minus_min`, `f2_ctx_minus_min`, `f2_dist_rank`, `f2_n_cand`), computed by `person2/experiments/context_features.py` from the raw `.track` and `.nodes` files and the candidate list only (no labels, no reference route), so the API computes them live with the same function. The change from the 5 frozen features was agreed by the team and every model was re-run on the new table (see 3.1.1).

**Metric definitions.** *Macro F1* averages the F1 of class 0 and class 1 at threshold 0.5. *Top-1 accuracy*: for each GPS point take the candidate with the highest probability; it is correct if that candidate carries label 1 (ties go to the smaller arc id). *Candidate miss rate*: share of GPS points whose reference arc is not among the candidates, which caps Top-1. *Log loss* penalises confident wrong probabilities. *ECE* is the average gap between predicted probability and observed frequency over 10 bins. *Route continuity (classifier-only)*: share of consecutive top picks that are connected in the road network, from `evaluation/route_metrics.py` (Person 3), before any HMM decoding.

| Model | Algorithm family | Key configuration | CV macro F1 (mean ± std) | CV Top-1 | Test Top-1 | Route continuity (classifier-only, test) | Test log loss | Test macro F1 | Inference time |
|---|---|---|---|---|---|---|---|---|---|
| xgboost | Boosting (gradient-boosted decision trees) | colsample_bytree=0.8, gamma=0, learning_rate=0.1, max_depth=6, min_child_weight=1, n_estimators=300, reg_lambda=1, subsample=0.8 | 0.9071 ± 0.0334 | 0.7640 | 0.8466 | 0.9340 | 0.0571 | 0.9227 | 16.6 ms per trip |
| gaussian_nb | Probabilistic (generative, Naive Bayes) | var_smoothing=1e-09 | 0.6989 ± 0.0205 | 0.6882 | 0.7443 | 0.8949 | 0.7344 | 0.6923 | 5.0 ms per trip |
| logistic_regression | Linear (logistic regression) | class_weight=balanced, max_iter=1000 | 0.7225 ± 0.0421 | 0.7256 | 0.8093 | 0.9151 | 0.2311 | 0.7452 | 2.6 ms per trip |
| random_forest | Tree ensemble (bagging) | class_weight=balanced_subsample, max_depth=12, n_estimators=100 | 0.8678 ± 0.0379 | 0.7647 | 0.8444 | 0.9326 | 0.1020 | 0.8844 | 22.3 ms per trip |
| hist_gradient_boosting | Boosting (histogram gradient boosting) | class_weight=balanced, learning_rate=0.05, max_iter=150 | 0.8513 ± 0.0616 | 0.7641 | 0.8449 | 0.9287 | 0.1079 | 0.8797 | 36.3 ms per trip |
| linear_svm | Kernel-based (linear SVM) | C=1, class_weight=balanced, max_iter=5000 | 0.6965 ± 0.0503 | 0.7309 | 0.8176 | 0.9132 | 0.3586 | 0.7202 | 2.3 ms per trip |

Reference point: predicting the base rate for every candidate gives a student-test log loss of 0.2469. Inference time is the median of 5 repeated `predict_proba` calls on the complete student-test candidate set after a warm-up, all models timed one after another on an otherwise idle machine (`person2/experiments/time_models.py`).

**Reading the table.**

- XGBoost is best on every probability and classification measure (student-test log loss 0.0571, macro F1 0.9227, CV macro F1 0.9071).
- Random forest and HGB are close behind on Top-1 (0.8444 and 0.8449 against 0.8466) but have about twice the log loss (0.1020, 0.1079). They are trained with `class_weight="balanced"`, which pushes positive probabilities up; XGBoost is trained without class weights and is the best calibrated (ECE 0.004 against 0.043 and 0.043).
- Gaussian NB has the worst log loss (0.7344, worse than the base rate 0.2469): several context features are very skewed (skew of `f2_end_dist` 17, `feat_time_gap` 260), which breaks its bell-curve assumption.
- XGBoost's CV Top-1 is lowest on fold 0 (0.491), the fold with the highest candidate miss rate (26.3%); among points whose reference arc is a candidate it ranges from 0.666 to 0.918. The student-test trips have a lower miss rate (2.5%).

### 3.1.1 Why the feature set changed (evidence)

With the 5 frozen features, the tuned XGBoost picked a wrong arc for 24.9% of student-test GPS points (classifier only). Of those wrong picks, 48.1% were an *adjacent* arc (sharing a node with the true arc: the point is near a junction), 29.7% were the *reverse-direction twin* of the true arc, and 22.2% were other arcs. The frozen features carry no information about where along an arc a point projects, use a one-step heading, and score each candidate without looking at the other candidates or at the neighbouring GPS points. With the frozen features the HMM decoder gained nothing from the classifier: on the student-test trips it scored the same with classifier emissions as with distance alone (section 5). The context features target exactly these cases. With them, XGBoost's wrong picks fall to 12.8% (25.9% adjacent, 45.0% reverse, 29.1% other).

| Model | Test Top-1, 5 frozen features | Test Top-1, with context features | Test log loss, 5 frozen | Test log loss, with context |
|---|---|---|---|---|
| xgboost | 0.7216 | 0.8466 | 0.1085 | 0.0571 |
| gaussian_nb | 0.5185 | 0.7443 | 0.4293 | 0.7344 |
| logistic_regression | 0.6603 | 0.8093 | 0.4299 | 0.2311 |
| random_forest | 0.7224 | 0.8444 | 0.2257 | 0.1020 |
| hist_gradient_boosting | 0.7246 | 0.8449 | 0.2359 | 0.1079 |
| linear_svm | 0.6404 | 0.8176 | 0.5560 | 0.3586 |

The 5-feature runs stay in `person2/experiments/logs/` and their tables and figures in `person2/experiments/results/v1_frozen_features/`.

**Primary selection metric.** We select on cross-validated log loss, with Top-1 as tie-break. The HMM decoder uses these probabilities as emission scores, so probability quality is what it needs; macro F1 depends on a fixed 0.5 threshold that is arbitrary when each GPS point has about 12 candidates and one positive (calibrated Gaussian NB, for example, changes log loss from 0.7344 to 0.1208 while its macro F1 moves from 0.6923 to 0.8084). On this basis XGBoost is the selected model.

### Probability calibration check

| Run | CV log loss (pooled) | Test log loss | Test ECE | Test macro F1 | Test Top-1 |
|---|---|---|---|---|---|
| XGBoost | 0.0867 | 0.0571 | 0.0044 | 0.9227 | 0.8466 |
| XGBoost + isotonic | 0.0792 | 0.0555 | 0.0048 | 0.9230 | 0.8468 |
| Gaussian NB | 0.7568 | 0.7344 | 0.1360 | 0.6923 | 0.7443 |
| Gaussian NB + isotonic | 0.1445 | 0.1208 | 0.0092 | 0.8084 | 0.7269 |

Isotonic calibration (fitted on out-of-fold scores, trip-grouped) changes XGBoost's log loss only from 0.0571 to 0.0555, so the final model is uncalibrated XGBoost. It repairs Gaussian NB's probabilities (0.7344 to 0.1208).

**Provenance.** XGBoost and Gaussian NB are Person 2's. Logistic regression, random forest and HGB use the hyperparameters from the teammates' scripts (`experiments/train_model*.py`); the linear SVM is Person 3's (`person2/experiments/models.py`). All were run through the same harness on the same table. KNN and the MLP were not re-run on the context-feature table.

## 3.2 Model subsections

### XGBoost (gradient-boosted decision trees)

**A. Model understanding**

1. *Why suitable.* Each candidate is described by 13 numeric clues whose effect interacts (how far a candidate may be from the GPS point depends on where along the arc it projects, on the heading agreement and on the neighbouring points). Boosted trees learn such interactions, need no scaling, and are insensitive to heavy tails (`feat_time_gap` reaches 153,064 s). The data support a non-linear model: on the same features XGBoost reaches log loss 0.0571 and macro F1 0.9227, logistic regression 0.2311 and 0.7452. With the logistic objective it outputs probabilities the HMM uses directly.
2. *Strengths and limitations.* Strengths: best log loss, best macro F1, best calibration (ECE 0.004), fit in 22 s on 2.5 million rows. Limitations: it scores each candidate alone, so it cannot guarantee a connected route (classifier-only Top-1 0.8466); at threshold 0.5 it finds 85.6% of true matches (precision 85.6%), so the route comes from ranking plus the HMM.

**B. Training and configuration**

5. *Important hyperparameters (initial values).* n_estimators = 300, max_depth = 6, learning_rate = 0.1, subsample = 0.8, colsample_bytree = 0.8, min_child_weight = 1, reg_lambda = 1.0, gamma = 0.0; histogram tree method, seed 42, no class weights. Tuned values: section 4.
6. *Objective.* Binary logistic loss (log loss), minimised by gradient boosting with L2 leaf regularisation (reg_lambda) and a split penalty (gamma).

**C. Experimental results**

7–8. *Student-test results (10 trips).* Top-1 0.8466 (0.8686 among points whose reference arc is a candidate), macro F1 0.9227 (macro precision 0.9229, macro recall 0.9226), log loss 0.0571, AUC 0.9840. Confusion matrix at 0.5: TN 255,607, FP 2,681, FN 2,698, TP 15,976. Cross-validation: macro F1 0.9071 ± 0.0334, log loss 0.0866.
9. *Overfitting or underfitting?* The loss curve (`xgboost_learning_curve.png`) shows the mean validation log loss lowest at round 34 (0.0755), ending at 0.0866 after 300 rounds while training loss falls to 0.0372 (final gap 0.0494). Validation loss rises 15% after its best round, which is clear overfitting in the number of rounds; the tuned settings (lower learning rate, more regularisation) remove most of it (section 4). Generalisation to unseen trips is sound: CV macro F1 0.9071, student-test 0.9227.

### Gaussian Naive Bayes

**A. Model understanding**

1. *Why suitable.* The simplest probabilistic classifier: one bell curve per feature and class, combined with Bayes' rule, fitted in closed form in under a second. It is the baseline that shows how much the flexible models gain.
2. *Strengths and limitations.* Strength: speed (0.8 s to train). Limitations on this data: the bell-curve assumption is badly wrong (skew 8.2 for `feat_dist_to_arc`, 17 for `f2_end_dist`, 260 for `feat_time_gap`), and the context features are strongly related to each other by construction (e.g. `f2_dist_minus_min` is derived from `feat_dist_to_arc`), so independence does not hold. Its raw log loss (0.7344) is worse than the base rate.

**B. Training and configuration**

5. *Hyperparameters.* `var_smoothing` = 1e-09; class priors from the data. Features are standardised, which does not change a Gaussian NB except through the floor.
6. *Objective.* Maximum likelihood under the Gaussian-independence model, solved in closed form; no iterative loss.

**C. Experimental results**

7–8. *Student-test results.* Top-1 0.7443, macro F1 0.6923, log loss 0.7344, AUC 0.9530. CV macro F1 0.6989 ± 0.0205. With isotonic calibration: log loss 0.1208, macro F1 0.8084, Top-1 0.7269.
9. *Overfitting or underfitting?* Underfitting: in-sample macro F1 0.6961 is close to the CV value 0.6989, so the model is too rigid rather than memorising. It has no epoch-wise history, so no training curve is drawn.

## 3.3 Visualisations

All figures are in `person2/experiments/results/`, drawn from the logged runs only (5-feature versions in `results/v1_frozen_features/`).

- `cv_macro_f1.png`: CV macro F1 per model (bar = mean, whisker = std, dots = folds).
- `student_test_top1_macro_f1.png`: Student-test Top-1 accuracy and macro F1 per model.
- `log_loss.png`: CV and student-test log loss per model.
- `confusion_matrices.png`: Student-test confusion matrices at threshold 0.5.
- `reliability.png`: Calibration curves on the student-test trips.
- `xgboost_learning_curve.png`: XGBoost train vs validation log loss per boosting round (the only model with an epoch-wise history).
- `xgboost_learning_curve_tuned.png`: The same curve for the tuned XGBoost.
- `with_calibration/reliability.png`: Calibration curves including the isotonic variants.

## 4. Bayesian hyperparameter optimisation (XGBoost)

1. *Hyperparameters searched:* n_estimators, max_depth, learning_rate, subsample, colsample_bytree, min_child_weight, reg_lambda, gamma.
2. *Ranges and distributions* (fixed before tuning; reasons in `person2/tuning/search_space.json`):

| Hyperparameter | Type | Range | Scale |
|---|---|---|---|
| n_estimators | int | 100 to 500 step 50 | linear |
| max_depth | int | 3 to 10 | linear |
| learning_rate | float | 0.01 to 0.3 | log |
| subsample | float | 0.5 to 1.0 | linear |
| colsample_bytree | float | 0.5 to 1.0 | linear |
| min_child_weight | float | 1.0 to 20.0 | log |
| reg_lambda | float | 0.01 to 50.0 | log |
| gamma | float | 0.0 to 5.0 | linear |

3. *Objective:* mean over the 5 trip-grouped folds of the validation log loss (minimised), on the 80 training trips only. Why: Probability quality, which is what the HMM emission model uses. Chosen as the default objective after the baseline runs showed that macro F1 at a fixed 0.5 threshold is misleading here: each GPS point has ~12 candidates and exactly one positive (candidate positive rate ~7%), and a calibrated Gaussian NB predicted no positives at all (macro F1 0.48) although its log loss improved and its ranking (AUC) was unchanged. Decided before any tuning run, using training-partition CV and student-test baselines only.
4. *Trials:* 15 requested, 12 completed and 3 stopped early by the median pruner; Bayesian optimisation: Tree-structured Parzen Estimator (optuna.samplers.TPESampler), first 5 trials random; 25 minutes in total.
5. *Best values* (trial 6, CV log loss 0.0740): n_estimators = 300, max_depth = 9, learning_rate = 0.01118, subsample = 0.6628, colsample_bytree = 0.9547, min_child_weight = 4.152, reg_lambda = 0.1175, gamma = 3.451.
6. *Tuned vs untuned* (same table, folds, seed and metrics):

| Metric | Untuned (defaults) | Tuned | Difference |
|---|---|---|---|
| CV log loss (the objective) | 0.0866 | 0.0740 | -0.0126 |
| CV macro F1 | 0.9071 | 0.9085 | +0.0014 |
| Student-test Top-1 | 0.8466 | 0.8472 | +0.0006 |
| Student-test macro F1 | 0.9227 | 0.9236 | +0.0009 |
| Student-test log loss | 0.0571 | 0.0549 | -0.0022 |
| Inference time (ms per trip) | 16.6 | 26.2 | +9.5 |
| Training time (s) | 22.2 | 28.9 | +6.8 |

The tuned model is better on the optimised quantity: CV log loss 0.0866 to 0.0740 (-14.6%). On the student-test trips: log loss -0.0022, Top-1 +0.0006, macro F1 +0.0009 (10 trips: differences of a few thousandths are within noise). The final model uses the tuned settings, chosen on the CV objective only. Loss curves: default best validation round 34 of 300 (gap 0.049); tuned best round 298 of 300 (gap 0.028).

An earlier study on the 5 frozen features is kept in `person2/tuning/studies/xgboost_cv_log_loss/` for the record. Figures: `person2/tuning/studies/xgboost_v2_cv_log_loss/optimization_history.png`, `param_importance.png`; full trial log `trials.csv`.

## Final end-to-end results (full API pipeline)

The complete pipeline (`api/pipeline.py`: candidates, frozen and context features, XGBoost probabilities, tuned log-space Viterbi decoder, server-side scoring) was run over every trip by `person2/experiments/evaluate_pipeline.py`. Accuracy is the share of GPS points whose decoded arc equals the reference arc for that point (`evaluation/route_metrics.py`).

| Partition | Trips | Pooled accuracy | Mean trip accuracy | Mean route continuity | Execution mode |
|---|---|---|---|---|---|
| student_test | 10 | 0.8728 | 0.8914 | 0.9706 | model_plus_viterbi (10) |
| evaluator_test | 10 | 0.8716 | 0.8923 | 0.9448 | model_plus_viterbi (10) |

The evaluator partition was scored once, after every model, feature and decoder choice had been fixed on the training and student-test trips; nothing was tuned on it.
