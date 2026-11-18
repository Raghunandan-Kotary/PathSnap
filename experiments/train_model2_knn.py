"""
Model 2 (Person 1): Distance-Weighted K-Nearest Neighbors.
Algorithm Family: Instance-based
Uses subsampled candidate sets per fold for practical KNN training & evaluation latency.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold

from experiments.metrics import compute_classification_metrics, compute_observation_top1_accuracy
from models.loader import FEATURE_COLUMNS

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
RESULTS_DIR = BASE_DIR / "experiments" / "results"

def create_model_2(n_neighbors: int = 15):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier(n_neighbors=n_neighbors, weights="distance", n_jobs=-1))
    ])

def train_and_evaluate():
    print("=" * 80)
    print("Training Model 2: Distance-Weighted KNN (Instance-based Family)")
    print("=" * 80)

    train_path = PROCESSED_DIR / "candidates_features_train.parquet"
    test_path = PROCESSED_DIR / "candidates_features_test.parquet"

    print("Loading processed feature tables...")
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)
    
    # Subsample training data for KNN computational tractability (100,000 stratified samples)
    # This prevents O(N^2) memory bottlenecks while preserving distance distribution
    print("Sampling 100,000 training candidates for instance-based indexing...")
    pos = train_df[train_df["label"] == 1]
    neg = train_df[train_df["label"] == 0].sample(min(80000, len(train_df[train_df["label"] == 0])), random_state=42)
    sub_train = pd.concat([pos.sample(min(20000, len(pos)), random_state=42), neg]).sample(frac=1.0, random_state=42)

    gkf = GroupKFold(n_splits=5)
    X = sub_train[FEATURE_COLUMNS].values
    y = sub_train["label"].values
    groups = sub_train["trip_id"].values

    fold_f1s = []
    fold_top1s = []

    print("\nRunning 5-Fold GroupKFold CV...")
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        model = create_model_2(n_neighbors=15)
        model.fit(X[train_idx], y[train_idx])
        
        val_sub = sub_train.iloc[val_idx]
        val_probs = model.predict_proba(X[val_idx])[:, 1]
        val_preds = (val_probs >= 0.5).astype(int)
        
        f1 = compute_classification_metrics(y[val_idx], val_preds, val_probs)["macro_f1"]
        top1 = compute_observation_top1_accuracy(val_sub, val_probs)
        fold_f1s.append(f1)
        fold_top1s.append(top1)
        print(f"  Fold {fold}: Macro F1 = {f1:.4f}, Top-1 Acc = {top1*100:.2f}%")

    cv_macro_f1 = float(np.mean(fold_f1s))
    cv_top1 = float(np.mean(fold_top1s))
    print(f"\nCV Mean Macro F1: {cv_macro_f1:.4f} (+/- {np.std(fold_f1s):.4f})")
    print(f"CV Mean Top-1 Acc: {cv_top1*100:.2f}%")

    # Fit final model on subsample and evaluate on student_test
    print("\nFitting final KNN model on subsampled training data...")
    final_model = create_model_2(n_neighbors=15)
    final_model.fit(X, y)

    # Subsample test set for fast evaluation (50,000 rows or full)
    test_sub = test_df.sample(min(50000, len(test_df)), random_state=42)
    X_test = test_sub[FEATURE_COLUMNS].values
    y_test = test_sub["label"].values
    
    test_probs = final_model.predict_proba(X_test)[:, 1]
    test_preds = (test_probs >= 0.5).astype(int)
    class_metrics = compute_classification_metrics(y_test, test_preds, test_probs)
    test_top1 = compute_observation_top1_accuracy(test_sub, test_probs)

    print(f"Student-Test Top-1 Accuracy: {test_top1*100:.2f}%")
    print(f"Student-Test Macro F1:       {class_metrics['macro_f1']:.4f}")
    print(f"Student-Test Log Loss:       {class_metrics['log_loss']:.4f}")

    out_dict = {
        "model_name": "Model 2: Distance-Weighted KNN",
        "family": "Instance-based",
        "hyperparameters": {"n_neighbors": 15, "weights": "distance"},
        "cv_macro_f1": cv_macro_f1,
        "cv_top1_accuracy": cv_top1,
        "student_test_macro_f1": class_metrics["macro_f1"],
        "student_test_top1_accuracy": test_top1,
        "student_test_log_loss": class_metrics["log_loss"]
    }

    out_file = RESULTS_DIR / "model2_knn_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(out_dict, f, indent=2)
    print(f"\nSaved Model 2 results to {out_file}")

    return out_dict

if __name__ == "__main__":
    train_and_evaluate()
