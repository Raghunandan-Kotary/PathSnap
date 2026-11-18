# Person 2 track: evaluation harness, model comparison and Bayesian optimisation

Everything in this folder is self-contained: it adds files only under `person2/` and does not modify or import
any other part of the repository. It holds the shared evaluation harness, the experiments for XGBoost and Gaussian
Naive Bayes, the consolidated model comparison, the Bayesian hyperparameter optimisation, and the generated text for
report sections 3.1, 3.2 (XGBoost, Gaussian NB), 3.3 and 4.

Report text: `person2/docs/report/PERSON2_REPORT_SECTIONS.md` (generated from the logs by `experiments/make_report_sections.py`).
Teammates who want to run their own model through the same harness: `person2/docs/TEAMMATE_GUIDE.md`.

## Folder map

| Path (under `person2/`) | Contents |
|---|---|
| `config.py`, `geo.py` | Paths (`PATHSNAP_DATA`, default `data/raw`), global seed 42, per-trip local projection used for every distance |
| `eda/` | `run_eda.py` (inventory, GPS quality, network connectivity, route checks), `make_eda_plots.py` (figures, Folium maps), `outputs/` |
| `experiments/harness.py`, `metrics.py`, `calibration.py` | Trip-grouped CV, student-test scoring, metric definitions, leak-free calibration, per-run logging |
| `experiments/models.py` | XGBoost, Gaussian NB, and the teammates' models with the hyperparameters from their own scripts |
| `experiments/adapt_team_table.py` | Validates Person 1's frozen tables and adds the `partition` column and the observation table |
| `experiments/run_person2_models.py`, `run_teammate_models.py`, `run_model.py` | Train and evaluate models through the harness |
| `experiments/compare.py`, `time_models.py`, `learning_curves.py` | Comparison table and plots, consistent timing, XGBoost loss curve |
| `experiments/make_report_sections.py` | Generates the report text from logged runs; stops if a sentence would contradict the logs |
| `experiments/logs/<run_id>/` | One folder per run: `config.json`, `metrics.json`, `fold_metrics.csv` |
| `experiments/results/` | Comparison table, figures, timing, loss curves |
| `experiments/tests/` | Hand-computed checks of the metrics, schema guards and the model loader |
| `tuning/` | `search_space.json`, `run_optuna.py`, `studies/xgboost_cv_log_loss/` (full trial log and figures) |
| `models/` | `save_final.py`, `load.py` (promote a logged run to the final model; load it in the API) |

## Install
Python 3.11. `pip install -r person2/requirements.txt`. The raw data is not committed: put `train/` and `student_test/`
in one directory and set `PATHSNAP_DATA` (default `data/raw`). The evaluator partition is never read by any code here.

## Reproduce, in order (run from the repository root)
```
python person2/eda/run_eda.py
python person2/eda/make_eda_plots.py
python -m features.build_features                      # Person 1's frozen table, commit 86ad172
python person2/experiments/adapt_team_table.py --train data/processed/candidates_features_train.parquet --test data/processed/candidates_features_test.parquet --out-dir person2/experiments/team_table
python person2/experiments/tests/test_harness.py
python person2/experiments/run_person2_models.py --table person2/experiments/team_table/candidates_features.parquet --obs person2/experiments/team_table/observations.parquet --features feat_dist_to_arc feat_arc_length feat_heading_diff feat_speed feat_time_gap
python person2/experiments/run_teammate_models.py --table ... --obs ... --features ... --models logistic_regression random_forest hist_gradient_boosting
python person2/tuning/run_optuna.py --model xgboost --table ... --obs ... --features ... --n-trials 15 --n-startup-trials 5
python person2/experiments/time_models.py --table ... --features ... --tuned-params person2/tuning/studies/xgboost_cv_log_loss/best_params.json
python person2/experiments/learning_curves.py --table ... --features ...
python person2/experiments/compare.py --table ...
python person2/experiments/make_report_sections.py --table ...
```
Model files and per-candidate probability files are large and are not committed (`.gitignore`); regenerate them with the runs above.

## Status
| Item | State |
|---|---|
| EDA, harness, tests | done |
| Context features (`experiments/context_features.py`) | done; final feature set = 5 frozen + 8 context features |
| XGBoost, Gaussian NB, logistic regression, random forest, HGB, linear SVM on the final table | done, `experiments/logs/` |
| Calibration variants (isotonic, Platt) | done |
| Comparison table, plots, loss curves, timing | done, `experiments/results/` (5-feature versions in `results/v1_frozen_features/`) |
| Bayesian optimisation of XGBoost (15 trials) | done, `tuning/studies/xgboost_v2_cv_log_loss/` |
| Final model | `models/final/model.joblib` (tuned XGBoost, 13 features), loaded by the API |
| Full pipeline on student-test and evaluator trips | done, `experiments/results/final_pipeline_*.json` |
| Report text 3.1, 3.2, 3.3, 4 | generated, `docs/report/PERSON2_REPORT_SECTIONS.md`; master report built by `docs/build_master_report.py` |

## Reproduce the final results
```
python person2/experiments/adapt_team_table.py --train data/processed/candidates_features_train.parquet --test data/processed/candidates_features_test.parquet --out-dir person2/experiments/team_table
python person2/experiments/context_features.py --table person2/experiments/team_table/candidates_features.parquet --data-dir data/raw --out person2/experiments/team_table_v2/candidates_features.parquet
python person2/experiments/run_teammate_models.py --table person2/experiments/team_table_v2/candidates_features.parquet --obs person2/experiments/team_table_v2/observations.parquet --features <13 columns> --models xgboost gaussian_nb logistic_regression random_forest hist_gradient_boosting linear_svm
python person2/tuning/run_optuna.py --model xgboost --table ... --obs ... --features <13 columns> --n-trials 15 --n-startup-trials 5 --out-dir person2/tuning/studies/xgboost_v2_cv_log_loss
python person2/models/save_final.py --run-dir person2/experiments/logs/<tuned run> --out-dir person2/models/final
python hmm/tune_decoder.py --table person2/experiments/team_table_v2/candidates_features.parquet --proba-dir person2/experiments/logs/<tuned run> --write-config
python person2/experiments/evaluate_pipeline.py --partition student_test
python person2/experiments/make_report_sections.py --v1-proba <proba_student_test.parquet of the 5-feature tuned run>
python person2/docs/build_master_report.py --pdf
```

## Integrity notes
* Cross-validation and the Optuna search use the 80 training trips only (`cross_validate` asserts this). Student-test trips are used afterwards to compare models. No evaluator data exists in this code.
* Preprocessing sits inside each model pipeline and is fitted on training folds only.
* The loader reads only the saved feature columns, so a `label` column cannot influence a prediction (tested).
* No predictions are hard-coded. Every number in the report text is generated from a logged run, with seeds, parameters, library versions and the feature-table hash recorded.
* Results published elsewhere in the repository for KNN, RF and MLP were computed on class-rebalanced random subsamples and are not comparable with the tables here.
