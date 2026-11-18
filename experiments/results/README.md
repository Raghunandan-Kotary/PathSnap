# experiments/results

- `person3_hmm_comparison_summary.json`, `person3_hmm_comparison_per_trip.csv`: HMM vs classifier-only results of the deployed decoder (`hmm/tune_decoder.py`).
- `dataset_inventory.csv`, `candidate_miss_rates.csv`, `observation_level_summary.csv`: dataset and candidate audit (Person 1).
- `model*_results.json`, `model_comparison_table.csv`: early experiments by Person 1 on the 5 frozen features, trained and cross-validated on class-rebalanced samples. They are kept as a record of those experiments; the model comparison reported in the technical report is in `person2/experiments/results/` (all models on the same table, complete validation folds).
