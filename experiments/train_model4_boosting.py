"""
Model 4 (Person 2): Histogram-based Gradient Boosting / LightGBM Classifier.
Algorithm Family: Boosting
Trains using shared evaluation harness on 80 training trips (5-fold group CV)
and evaluates on 10 student-test trips.
"""
import json
from pathlib import Path
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from experiments.harness import ModelEvaluationHarness
from models.loader import FEATURE_COLUMNS

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
RESULTS_DIR = BASE_DIR / "experiments" / "results"

def create_model_4(learning_rate: float = 0.05, max_iter: int = 150):
    return HistGradientBoostingClassifier(
        learning_rate=learning_rate,
        max_iter=max_iter,
        class_weight="balanced",
        random_state=42
    )

def train_and_evaluate():
    print("=" * 80)
    print("Training Model 4: Gradient Boosting (Boosting Family)")
    print("=" * 80)

    train_path = PROCESSED_DIR / "candidates_features_train.parquet"
    test_path = PROCESSED_DIR / "candidates_features_test.parquet"

    print("Loading processed feature tables...")
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)
    print(f"Loaded: {len(train_df):,} train rows, {len(test_df):,} test rows.")

    harness = ModelEvaluationHarness(n_splits=5, random_seed=42)

    # 1. 5-Fold GroupKFold CV
    print("\nRunning 5-Fold GroupKFold Cross-Validation on 80 training trips...")
    cv_res = harness.run_group_cv("Gradient_Boosting", create_model_4, train_df)
    print(f"  CV Macro F1: {cv_res['cv_macro_f1_mean']:.4f} (+/- {cv_res['cv_macro_f1_std']:.4f})")
    print(f"  CV Top-1 Acc: {cv_res['cv_top1_acc_mean']*100:.2f}%")
    print(f"  CV Training Time: {cv_res['training_time_s']:.2f}s")

    # 2. Fit on full training set and evaluate on student_test
    print("\nRefitting on full training set and evaluating on 10 student-test trips...")
    final_model = create_model_4()
    X_train = train_df[FEATURE_COLUMNS].values
    y_train = train_df["label"].values
    final_model.fit(X_train, y_train)

    test_res = harness.evaluate_student_test("Gradient_Boosting", final_model, test_df)
    print(f"  Student-Test Top-1 Accuracy: {test_res['student_test_top1_acc']*100:.2f}%")
    print(f"  Student-Test Macro F1:       {test_res['student_test_macro_f1']:.4f}")
    print(f"  Student-Test Log Loss:       {test_res['student_test_log_loss']:.4f}")
    print(f"  Inference Latency:           {test_res['inference_time_ms']:.2f}ms")

    # 3. Save results
    out_dict = {
        "model_name": "Model 4: HistGradientBoosting",
        "family": "Boosting",
        "hyperparameters": {"learning_rate": 0.05, "max_iter": 150, "class_weight": "balanced"},
        "cv_results": cv_res,
        "student_test_results": test_res
    }

    out_file = RESULTS_DIR / "model4_boosting_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(out_dict, f, indent=2)
    print(f"\nSaved Model 4 results to {out_file}")

    return out_dict, final_model

if __name__ == "__main__":
    train_and_evaluate()
