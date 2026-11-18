# Person 2: report sections (generated from logged runs)

> Every number below is read from `experiments/logs/`, `tuning/studies/` or the frozen feature table by `experiments/make_report_sections.py`. Regenerate it after any new run.

## 3.1 Overall model comparison

**How every model was measured.** All models use the same frozen feature table (Person 1, commit `86ad172`): 2,546,809 candidate rows from 80 training trips (208,417 GPS points) and 276,962 candidate rows from the 10 student-test trips (19,160 GPS points). Each candidate is a (GPS point, road arc) pair; the label is 1 for the single arc the GPS point was matched to on the reference route. Positives are 7.5% of training candidates and 6.7% of student-test candidates, with 12.2 and 14.5 candidates per GPS point. The five features are `feat_dist_to_arc`, `feat_arc_length`, `feat_heading_diff`, `feat_speed` and `feat_time_gap`. Cross-validation uses the trip-grouped 5-fold split shipped in the table (16 trips per fold, no trip in two folds), seed 42. Validation folds and the student-test set are scored in full, with no resampling. All runs were trained and timed on one machine (16 logical CPUs).

**Metric definitions.** *Macro F1* averages the F1 of class 0 and class 1 at probability threshold 0.5. *Top-1 accuracy*: for each GPS point take the candidate with the highest probability; it is correct if that candidate carries label 1 (ties go to the smaller arc id). *Candidate miss rate*: share of GPS points whose reference arc is not among the candidates at all; no model can get those right. *Log loss* penalises confident wrong probabilities. *ECE* is the average gap between predicted probability and observed frequency over 10 bins.

| Model | Algorithm family | Key configuration | CV macro F1 (mean ± std) | CV Top-1 | Test Top-1 | Route continuity | Test log loss | Test macro F1 | Inference time |
|---|---|---|---|---|---|---|---|---|---|
| xgboost | Boosting (gradient-boosted decision trees) | colsample_bytree=0.8, gamma=0, learning_rate=0.1, max_depth=6, min_child_weight=1, n_estimators=300, reg_lambda=1, subsample=0.8 | 0.7935 ± 0.0233 | 0.6414 | 0.7216 | N/A: needs the decoded road sequence (reported in section 5) | 0.1085 | 0.8180 | 16.5 ms per trip (0.59 ms per 1,000 candidates) |
| gaussian_nb | Probabilistic (generative, Naive Bayes) | var_smoothing=1e-09 | 0.6004 ± 0.0207 | 0.4676 | 0.5185 | N/A: needs the decoded road sequence (reported in section 5) | 0.4293 | 0.6320 | 2.3 ms per trip (0.08 ms per 1,000 candidates) |
| logistic_regression | Linear (logistic regression) | class_weight=balanced, max_iter=1000 | 0.6457 ± 0.0240 | 0.5816 | 0.6603 | N/A: needs the decoded road sequence (reported in section 5) | 0.4299 | 0.6449 | 1.4 ms per trip (0.05 ms per 1,000 candidates) |
| random_forest | Tree ensemble (bagging) | class_weight=balanced_subsample, max_depth=12, n_estimators=100 | 0.7463 ± 0.0083 | 0.6396 | 0.7224 | N/A: needs the decoded road sequence (reported in section 5) | 0.2257 | 0.7526 | 23.3 ms per trip (0.84 ms per 1,000 candidates) |
| hist_gradient_boosting | Boosting (histogram gradient boosting) | class_weight=balanced, learning_rate=0.05, max_iter=150 | 0.7348 ± 0.0053 | 0.6388 | 0.7246 | N/A: needs the decoded road sequence (reported in section 5) | 0.2359 | 0.7417 | 32.1 ms per trip (1.16 ms per 1,000 candidates) |

Inference time is the median of 5 repeated `predict_proba` calls on the complete student-test candidate set (276,962 rows, 10 trips) after a warm-up call; all models were timed one after another on an otherwise idle machine (`experiments/time_models.py`). Fit time is one fit on all 80 training trips.

Reference point: predicting the base rate for every candidate gives a student-test log loss of 0.2469.

**Reading the table.**

- XGBoost is best on probability quality and on macro F1 (student-test log loss 0.1085, macro F1 0.8180; CV macro F1 0.7935).
- Top-1 does not separate the three tree models: XGBoost 0.7216, random forest 0.7224, HGB 0.7246 (spread 0.30 points on 10 trips, not a meaningful difference).
- Gaussian NB (0.4293) and logistic regression (0.4299) have a log loss worse than the base-rate reference (0.2469), i.e. worse than giving every candidate the same base-rate probability. The logistic regression, random forest and HGB use `class_weight="balanced"`, which pushes positive probabilities up; their reliability curves sit below the diagonal (`reliability.png`), while XGBoost, trained without class weights, follows the diagonal (ECE 0.004 against 0.109, 0.113 and 0.262).
- CV Top-1 for XGBoost ranges from 0.444 (fold 0) to 0.741 (fold 4) because the candidate miss rate differs by fold, from 1.5% to 26.3%. Fold 0 contains trip 00000054 (19,313 GPS points, 51% of them without the reference arc among the candidates). Among points whose reference arc is a candidate, XGBoost's Top-1 ranges only from 0.602 to 0.752. The student-test trips have a lower miss rate (2.5%), which is why student-test Top-1 (0.7216) is higher than CV Top-1 (0.6414).

**Primary selection metric.** We select on cross-validated log loss, with Top-1 among points that have a reference candidate as tie-break. Reasons: (1) the HMM decoder uses these probabilities as emission scores, so probability quality is what it needs; (2) macro F1 depends on a fixed 0.5 threshold, and with about 12 candidates per point and one positive, that threshold is arbitrary: isotonic-calibrated Gaussian NB has a better log loss (0.1860 against 0.4293) and the same ranking quality (AUC 0.8626 against 0.8648) but predicts no positives at threshold 0.5, so its macro F1 drops to 0.4826; (3) Top-1 is capped by the candidate miss rate and is nearly equal across the strong models. Macro F1 is still reported because the template requires it. On this basis XGBoost is the recommended model for tuning; the final model choice is made by the team from this table.

### Probability calibration check

| Run | CV log loss (pooled) | Test log loss | Test ECE | Test macro F1 | Test Top-1 | Positives predicted at 0.5 (test) |
|---|---|---|---|---|---|---|
| XGBoost | 0.1349 | 0.1085 | 0.0042 | 0.8180 | 0.7216 | 15,024 |
| XGBoost + isotonic | 0.1333 | 0.1091 | 0.0059 | 0.8162 | 0.7207 | 14,587 |
| Gaussian NB | 0.3794 | 0.4293 | 0.0473 | 0.6320 | 0.5185 | 9,527 |
| Gaussian NB + isotonic | 0.2118 | 0.1860 | 0.0234 | 0.4826 | 0.4851 | 0 |

Isotonic calibration (fitted on out-of-fold scores, trip-grouped) changes XGBoost's log loss by 0.0006, so XGBoost needs no calibration. It cuts Gaussian NB's log loss from 0.4293 to 0.1860, but isotonic regression outputs a step function: the share of GPS points whose two best candidates have identical probability rises from 6.5% to 31.3% (0.0% for XGBoost); ties are resolved by the smaller arc id, which carries no information, so Top-1 falls (0.5185 to 0.4851). It also leaves no candidate above 0.5.

## 3.2 Model subsections

### XGBoost (gradient-boosted decision trees)

**A. Model understanding**

1. *Why suitable.* Each candidate is described by five numeric clues, and how much a clue such as distance matters plausibly depends on the others (arc length, heading, speed). Boosted trees can learn such interactions, and the data support that the relationship is not linear: XGBoost reaches log loss 0.1085 and macro F1 0.8180 where logistic regression, a linear model on the same features, reaches 0.4299 and 0.6449. Trees also need no feature scaling and are insensitive to the heavy tails in our features (`feat_time_gap` is exactly 1 s for 96.4% of training rows but reaches 153,064 s; `feat_dist_to_arc` has skew 8.2), because trees only use the order of values. Trained with the logistic objective, it also outputs probabilities that the HMM can use directly.
2. *Strengths and limitations.* Strengths seen in our runs: best log loss (0.1085), best macro F1 (0.8180), almost perfect calibration (ECE 0.004), and a fit time of 18 s on 2.5 million rows. Limitations: (a) with the default 300 rounds it overfits mildly (see C9); (b) at threshold 0.5 it finds only 59.4% of true matches (precision 73.8%), so route quality has to come from ranking plus the HMM, not from thresholding; (c) it scores each candidate alone, so on its own it cannot guarantee a connected route: Top-1 is 0.7216, not 1.

**B. Training and configuration**

5. *Important hyperparameters (initial values).* n_estimators = 300, max_depth = 6, learning_rate = 0.1, subsample = 0.8, colsample_bytree = 0.8, min_child_weight = 1, reg_lambda = 1.0, gamma = 0.0; histogram tree method, seed 42, no class weights.
6. *Objective.* Binary logistic loss (log loss), minimised by gradient boosting with L2 leaf regularisation (reg_lambda) and a split penalty (gamma).

**C. Experimental results**

7–8. *Student-test results (10 trips).* Top-1 accuracy 0.7216 (0.7403 among points whose reference arc is a candidate; candidate miss rate 2.5%), macro F1 0.8180 (macro precision 0.8545, macro recall 0.7893), log loss 0.1085, AUC 0.9603. Confusion matrix at 0.5: TN 254,352, FP 3,936, FN 7,586, TP 11,088. Cross-validation: macro F1 0.7935 ± 0.0233, log loss 0.1349.
9. *Overfitting or underfitting?* Mild overfitting in the number of rounds, no underfitting. In the loss curve the mean validation log loss is lowest at round 47 (0.1305) and then creeps up to 0.1349 at round 300, while training loss keeps falling to 0.1091 (final gap 0.0258). The cost of the extra rounds is small (3.4% in log loss). Generalisation to unseen trips is sound: CV macro F1 0.7935 against student-test 0.8180; the test trips are easier because fewer points lack a reference candidate (2.5%).

### Gaussian Naive Bayes

**A. Model understanding**

1. *Why suitable.* It is the simplest probabilistic classifier: for each class it fits one bell curve per feature and combines them with Bayes' rule. It trains in closed form in under a second, which makes it a useful baseline that shows how much the other models gain from flexible decision boundaries.
2. *Strengths and limitations.* Strength: speed (0.3 s to train, 2.3 ms per trip to score). Limitations on this data: the bell-curve assumption is badly wrong (skew of `feat_dist_to_arc` 8.2, of `feat_arc_length` 10.2, of `feat_time_gap` 260; `feat_time_gap` equals 1 s in 96.4% of rows), and independence holds only roughly (largest correlation between two features 0.13 overall, 0.46 among true matches). Its raw probabilities are poor: log loss 0.4293 is worse than the base-rate reference 0.2469.

**B. Training and configuration**

5. *Hyperparameters.* `var_smoothing` = 1e-09 (a tiny variance floor), class priors estimated from the data (7.5% positive); features are standardised, which does not change a Gaussian NB except through the floor.
6. *Objective.* Maximum likelihood of the training data under the Gaussian-independence model, solved in closed form (means and variances per class and feature); there is no iterative loss.

**C. Experimental results**

7–8. *Student-test results.* Top-1 0.5185, macro F1 0.6320 (positive-class precision 0.4462, recall 0.2276), log loss 0.4293, AUC 0.8648, ECE 0.0473. Cross-validation: macro F1 0.6004 ± 0.0207.
9. *Overfitting or underfitting?* Underfitting (high bias). The in-sample macro F1 on the training data is 0.6124, almost identical to the cross-validated 0.6004 and the student-test 0.6320: the model is not memorising anything, it is simply too rigid for this data. Gaussian NB has no epoch-wise history, so no training curve is drawn. Isotonic calibration improves log loss to 0.1860, but then no candidate passes 0.5 (macro F1 0.4826).

**Provenance of the models in the table.** XGBoost and Gaussian NB are Person 2's. Logistic regression, random forest and HGB were run here, through the same harness, with the hyperparameters from the teammates' scripts in the team repository (`experiments/train_model*.py`), so that every row uses the same folds and complete validation sets. Their owners explain those models in their own subsections. KNN and the MLP are still to be added by their owners through the same harness; the results published in the team repository for those two were computed on class-rebalanced random subsamples and are not comparable with this table.

## 3.3 Visualisations

All figures are in `experiments/results/` and are drawn from the logged runs only.

- `cv_macro_f1.png`: CV macro F1 per model (bar = mean, whisker = std, dots = the 5 folds).
- `student_test_top1_macro_f1.png`: Student-test Top-1 accuracy and macro F1 per model.
- `log_loss.png`: CV and student-test log loss per model.
- `confusion_matrices.png`: Student-test confusion matrices at threshold 0.5 (optional diagnostic).
- `reliability.png`: Calibration curves on the student-test trips; the dashed diagonal is perfect calibration.
- `xgboost_learning_curve.png`: XGBoost train vs validation log loss per boosting round, untuned defaults. XGBoost is the only model here with an epoch-wise history, so no other model gets a loss curve.
- `xgboost_learning_curve_tuned.png`: The same curve for the tuned XGBoost.
- `with_calibration/reliability.png`: Calibration curves including the isotonic variants.
- Folium maps (raw GPS, road network, reference route, predicted route) are produced by the decoder track (section 6); `eda/outputs/folium_train_00000003.html` shows a trip with GPS, road network and reference route.

## 4. Bayesian hyperparameter optimisation (XGBoost)

1. *Hyperparameters searched:* n_estimators, max_depth, learning_rate, subsample, colsample_bytree, min_child_weight, reg_lambda, gamma.
2. *Ranges and distributions* (fixed before any tuning run; see `tuning/search_space.json` for the reasoning behind each):

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

3. *Objective:* cv_log_loss = mean over the 5 trip-grouped folds of the validation log loss (minimize); the search uses the 80 training trips only. Why this objective: Probability quality, which is what the HMM emission model uses. Chosen as the default objective after the baseline runs showed that macro F1 at a fixed 0.5 threshold is misleading here: each GPS point has ~12 candidates and exactly one positive (candidate positive rate ~7%), and a calibrated Gaussian NB predicted no positives at all (macro F1 0.48) although its log loss improved and its ranking (AUC) was unchanged. Decided before any tuning run, using training-partition CV and student-test baselines only.
4. *Trials:* 15 requested, 9 completed and 6 stopped early by the median pruner. Sampler: Bayesian optimisation: Tree-structured Parzen Estimator (optuna.samplers.TPESampler); the first 5 trials are random. Total optimisation time 20 minutes of wall-clock time (81 s per trial on average; the machine was shared with other jobs during part of the search, so single-trial times are not clean timings).
5. *Best values* (trial 10, CV log loss 0.1310): n_estimators = 450, max_depth = 6, learning_rate = 0.01019, subsample = 0.523, colsample_bytree = 0.9506, min_child_weight = 1.251, reg_lambda = 0.01093, gamma = 1.184.
6. *Did the tuned model beat the untuned one?* Same table, folds, seed and metrics for both:

| Metric | Untuned (defaults) | Tuned | Difference |
|---|---|---|---|
| CV log loss (the tuning objective) | 0.1349 | 0.1310 | -0.0040 |
| CV macro F1 | 0.7935 | 0.7920 | -0.0016 |
| Student-test Top-1 | 0.7216 | 0.7253 | +0.0037 |
| Student-test macro F1 | 0.8180 | 0.8157 | -0.0023 |
| Student-test log loss | 0.1085 | 0.1080 | -0.0005 |
| Inference time (ms per trip) | 16.5 | 26.7 | +10.2 |
| Training time (s) | 18.5 | 31.9 | +13.4 |

On the quantity that was optimised, the tuned model is better: CV log loss 0.1349 to 0.1310 (-2.9%). On the student-test trips the change is negligible: log loss -0.0005, Top-1 +0.0037, macro F1 -0.0023; with only 10 test trips, differences of a few thousandths are within noise. So tuning did not change which model is best or what the decoder will receive in practice. The tuned model is also slower (fit 18 s to 32 s, inference 16.5 to 26.7 ms per trip). The loss curves show why it helps a little: the default (learning rate 0.1) reaches its lowest validation loss at round 47 of 300 and then drifts upward (train-validation gap 0.026 at the end), whereas the tuned model (learning rate 0.010) keeps improving until round 370 of 450 and overfits less (gap 0.016).

Figures: `tuning/studies/xgboost_cv_log_loss/optimization_history.png`, `param_importance.png`. Full trial log: `trials.csv`.
