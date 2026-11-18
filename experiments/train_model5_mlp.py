"""
Model 5 (Person 2): Multi-Layer Perceptron (Neural Network) Classifier.
Algorithm Family: Neural Networks
Trains with StandardScaler and early stopping on stratified candidate sample.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold

from experiments.metrics import compute_classification_metrics, compute_observation_top1_accuracy
from models.loader import FEATURE_COLUMNS

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
RESULTS_DIR = BASE_DIR / "experiments" / "results"

def create_model_5(hidden_layer_sizes=(64, 32), max_iter: int = 100):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation="relu",
            max_iter=max_iter,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1
        ))
    ])

def train_and_evaluate():
    print("=" * 80)
    print("Training Model 5: Multi-Layer Perceptron (Neural Network Family)")
    print("=" * 80)

    train_path = PROCESSED_DIR / "candidates_features_train.parquet"
    test_path = PROCESSED_DIR / "candidates_features_test.parquet"

    print("Loading processed feature tables...")
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    # Subsample 150,000 candidate rows for MLP training stability
    print("Sampling 150,000 training candidates for neural network training...")
    pos = train_df[train_df["label"] == 1]
    neg = train_df[train_df["label"] == 0].sample(min(120000, len(train_df[train_df["label"] == 0])), random_state=42)
    sub_train = pd.concat([pos.sample(min(30000, len(pos)), random_state=42), neg]).sample(frac=1.0, random_state=42)

    gkf = GroupKFold(n_splits=5)
    X = sub_train[FEATURE_COLUMNS].values
    y = sub_train["label"].values
    groups = sub_train["trip_id"].values

    fold_f1s = []
    fold_top1s = []

    print("\nRunning 5-Fold GroupKFold CV...")
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        model = create_model_5(hidden_layer_sizes=(64, 32), max_iter=80)
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
    print("\nFitting final MLP model and evaluating on 10 student-test trips...")
    final_model = create_model_5(hidden_layer_sizes=(64, 32), max_iter=100)
    final_model.fit(X, y)

    X_test = test_df[FEATURE_COLUMNS].values
    y_test = test_df["label"].values

    test_probs = final_model.predict_proba(X_test)[:, 1]
    test_preds = (test_probs >= 0.5).astype(int)
    class_metrics = compute_classification_metrics(y_test, test_preds, test_probs)
    test_top1 = compute_observation_top1_accuracy(test_df, test_probs)

    print(f"  Student-Test Top-1 Accuracy: {test_top1*100:.2f}%")
    print(f"  Student-Test Macro F1:       {class_metrics['macro_f1']:.4f}")
    print(f"  Student-Test Log Loss:       {class_metrics['log_loss']:.4f}")

    out_dict = {
        "model_name": "Model 5: Multi-Layer Perceptron",
        "family": "Neural Network",
        "hyperparameters": {"hidden_layers": [64, 32], "activation": "relu", "early_stopping": True},
        "cv_macro_f1": cv_macro_f1,
        "cv_top1_accuracy": cv_top1,
        "student_test_macro_f1": class_metrics["macro_f1"],
        "student_test_top1_accuracy": test_top1,
        "student_test_log_loss": class_metrics["log_loss"]
    }

    out_file = RESULTS_DIR / "model5_mlp_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(out_dict, f, indent=2)
    print(f"\nSaved Model 5 results to {out_file}")

    return out_dict

if __name__ == "__main__":
    train_and_evaluate()
