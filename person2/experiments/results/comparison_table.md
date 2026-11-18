| Model | Algorithm family | Key configuration | CV macro F1 (mean ± std over folds) | Top-1 accuracy (student test) | Route continuity | Log loss (student test) | Macro F1 (student test) | Inference time (median of 5 repeats; ms per trip / per 1k candidates) |
|---|---|---|---|---|---|---|---|---|
| xgboost | Boosting (gradient-boosted decision trees) | colsample_bytree=0.8, gamma=0, learning_rate=0.1, max_depth=6, min_child_weight=1, n_estimators=300, reg_lambda=1, subsample=0.8 | 0.9071 ± 0.0334 | 0.8466 | 0.9340 | 0.0571 | 0.9227 | 16.6 / 0.60 |
| gaussian_nb | Probabilistic (generative, Naive Bayes) | var_smoothing=1e-09 | 0.6989 ± 0.0205 | 0.7443 | 0.8949 | 0.7344 | 0.6923 | 5.0 / 0.18 |
| logistic_regression | Linear (logistic regression) | class_weight=balanced, max_iter=1000 | 0.7225 ± 0.0421 | 0.8093 | 0.9151 | 0.2311 | 0.7452 | 2.6 / 0.10 |
| random_forest | Tree ensemble (bagging) | class_weight=balanced_subsample, max_depth=12, n_estimators=100 | 0.8678 ± 0.0379 | 0.8444 | 0.9326 | 0.1020 | 0.8844 | 22.3 / 0.81 |
| hist_gradient_boosting | Boosting (histogram gradient boosting) | class_weight=balanced, learning_rate=0.05, max_iter=150 | 0.8513 ± 0.0616 | 0.8449 | 0.9287 | 0.1079 | 0.8797 | 36.3 / 1.31 |
| linear_svm | Kernel-based (linear SVM) | C=1, class_weight=balanced, max_iter=5000 | 0.6965 ± 0.0503 | 0.8176 | 0.9132 | 0.3586 | 0.7202 | 2.2 / 0.08 |
