"""
Shared Evaluation Harness for PathSnap (Person 2 & Person 1).
Implements GroupKFold cross-validation on 80 training trips and student-test evaluation.
Logs configuration, random seed, training time, and inference time for every run.
"""
import time
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from experiments.metrics import compute_classification_metrics, compute_observation_top1_accuracy
from models.loader import FEATURE_COLUMNS

class ModelEvaluationHarness:
    def __init__(self, n_splits: int = 5, random_seed: int = 42):
        self.n_splits = n_splits
        self.random_seed = random_seed
        self.results_log: List[Dict[str, Any]] = []

    def run_group_cv(
        self,
        model_name: str,
        model_factory,
        train_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Executes trip-grouped 5-fold cross-validation.
        Ensures observations from the same trip never leak across train/val folds.
        """
        gkf = GroupKFold(n_splits=self.n_splits)
        groups = train_df["trip_id"].values
        X = train_df[FEATURE_COLUMNS].values
        y = train_df["label"].values

        fold_f1s = []
        fold_top1s = []
        t0 = time.perf_counter()

        for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
            model = model_factory()
            model.fit(X[train_idx], y[train_idx])
            
            val_sub = train_df.iloc[val_idx]
            val_probs = model.predict_proba(X[val_idx])[:, 1]
            val_preds = (val_probs >= 0.5).astype(int)
            
            f1 = compute_classification_metrics(y[val_idx], val_preds, val_probs)["macro_f1"]
            top1 = compute_observation_top1_accuracy(val_sub, val_probs)
            
            fold_f1s.append(f1)
            fold_top1s.append(top1)

        total_train_time = time.perf_counter() - t0

        cv_summary = {
            "model_name": model_name,
            "cv_macro_f1_mean": float(np.mean(fold_f1s)),
            "cv_macro_f1_std": float(np.std(fold_f1s)),
            "cv_top1_acc_mean": float(np.mean(fold_top1s)),
            "training_time_s": round(total_train_time, 3),
            "random_seed": self.random_seed
        }
        self.results_log.append(cv_summary)
        return cv_summary

    def evaluate_student_test(
        self,
        model_name: str,
        trained_model,
        test_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """Evaluates final model on 10 student-testing trips."""
        X_test = test_df[FEATURE_COLUMNS].values
        y_test = test_df["label"].values

        t_infer_start = time.perf_counter()
        probs = trained_model.predict_proba(X_test)[:, 1]
        inference_time_ms = (time.perf_counter() - t_infer_start) * 1000

        preds = (probs >= 0.5).astype(int)
        class_metrics = compute_classification_metrics(y_test, preds, probs)
        top1 = compute_observation_top1_accuracy(test_df, probs)

        return {
            "model_name": model_name,
            "student_test_top1_acc": top1,
            "student_test_macro_f1": class_metrics["macro_f1"],
            "student_test_log_loss": class_metrics["log_loss"],
            "inference_time_ms": round(inference_time_ms, 2)
        }
