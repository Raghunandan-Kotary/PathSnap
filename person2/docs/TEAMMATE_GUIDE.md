# Running your model through the shared harness

Why: so every model in report section 3.1 is measured the same way (same frozen table, same trip-grouped folds,
complete validation and test sets, one seed). Numbers from different protocols cannot be put in one table.

Nothing here edits your files or anyone else's. Everything lives in the `person2/` folder.

## 1. Install
```
pip install -r person2/requirements.txt
```

## 2. Prepare the table (once)
You need Person 1's two parquet files (`candidates_features_train.parquet`, `candidates_features_test.parquet`).
```
python person2/experiments/adapt_team_table.py --train <train.parquet> --test <test.parquet> --out-dir person2/experiments/team_table
```
It checks the table (no trip in two partitions or folds, no constant features, no duplicate candidates), adds the
`partition` column and builds the observation table. It should end with `no structural problems found`.

## 3. Run your models
Already registered with the hyperparameters from the team scripts: `logistic_regression`, `knn_distance`,
`random_forest`, `hist_gradient_boosting`, `mlp`, plus `xgboost` and `gaussian_nb`.
```
python person2/experiments/run_teammate_models.py ^
  --table person2/experiments/team_table/candidates_features.parquet ^
  --obs person2/experiments/team_table/observations.parquet ^
  --features feat_dist_to_arc feat_arc_length feat_heading_diff feat_speed feat_time_gap ^
  --models mlp knn_distance
```
(On macOS/Linux replace `^` with `\`.) On 16 cores expect about 25 minutes for `mlp` and 10 to 20 for `knn_distance`.

A model that is not registered (Linear SVM, Extra Trees, ...): add one entry to `person2/experiments/models.py`:
```python
def build_extra_trees(params, seed):
    return Pipeline([("scaler", "passthrough"), ("clf", ExtraTreesClassifier(random_state=seed, n_jobs=-1, **params))])
EXTRA_TREES = ModelSpec("extra_trees", "Tree ensemble (extra-randomised)", build_extra_trees, dict(n_estimators=100, max_depth=12))
```
and add it to `REGISTRY`. Preprocessing (scalers) must sit inside the pipeline so it is fitted on training folds only.
Train on the full training folds. If you must subsample for speed, subsample only the data used for fitting, never the validation folds or the test set.

## 4. Send back
The folders `person2/experiments/logs/<run_id>/` (each holds `config.json`, `metrics.json`, `fold_metrics.csv`; a few KB).
Person 2 rebuilds the comparison table and plots from them. The big `.parquet` files in those folders stay local.

## 5. Your report subsection
Answer these for each of your models (from `Deliverables.pdf`, section 3.2), using the numbers in your run folder:

| Question | Where the number comes from |
|---|---|
| A1 Why is it suitable for PathSnap? | your reasoning, tied to a result in the table |
| A2 Strengths and limitations | your results, e.g. calibration (`ece`), recall (`pos_recall`), timing |
| B5 Hyperparameters and initial values | `config.json` -> `params` |
| B6 Loss or objective | the algorithm; say what it minimises or maximises |
| C7, C8 Student-test Top-1, macro F1, log loss | `metrics.json` -> `student_test` -> `top1_acc`, `macro_f1`, `log_loss` |
| C9 Overfitting or underfitting? | training vs validation evidence: a loss curve (only if the model has epoch-wise history), or in-sample vs CV score |

Do not copy another model's explanations: every member can be asked about their own results in the viva.
