"""
Trains and evaluates Person 3's two models (Random Forest, calibrated
Linear SVM) through the shared harness: trip-grouped 5-fold CV on the 80
training trips, then a clean student-test evaluation.

Run from the repo root (needs data/processed/candidates_features_train.csv
and candidates_features_student_test.csv -- build them first with
experiments/build_training_table.py):
    python experiments/run_person3_training.py
"""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import pandas as pd

from experiments.harness import ModelEvaluationHarness
from experiments.train_model3_rf import create_model_3
from experiments.train_model_svm import create_model_svm
from models.loader import FEATURE_COLUMNS

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "processed"
RESULTS_DIR = REPO_ROOT / "experiments" / "results"
SAVED_DIR = REPO_ROOT / "models" / "saved"

MODELS = {
    "RandomForest": create_model_3,
    "CalibratedLinearSVM": create_model_svm,
}


def main():
    train_df = pd.read_csv(DATA_DIR / "candidates_features_train.csv", dtype={"trip_id": str})
    test_df = pd.read_csv(DATA_DIR / "candidates_features_student_test.csv", dtype={"trip_id": str})
    print(f"train: {len(train_df)} rows / {train_df['trip_id'].nunique()} trips, "
          f"student_test: {len(test_df)} rows / {test_df['trip_id'].nunique()} trips")

    harness = ModelEvaluationHarness(n_splits=5, random_seed=42)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    SAVED_DIR.mkdir(parents=True, exist_ok=True)

    comparison_rows = []
    for model_name, factory in MODELS.items():
        print(f"\n=== {model_name} ===")

        cv_summary = harness.run_group_cv(model_name, factory, train_df)
        print("CV:", cv_summary)

        final_model = factory()
        final_model.fit(train_df[FEATURE_COLUMNS].values, train_df["label"].values)
        test_summary = harness.evaluate_student_test(model_name, final_model, test_df)
        print("Student-test:", test_summary)

        model_path = SAVED_DIR / f"{model_name.lower()}.joblib"
        joblib.dump(final_model, model_path)
        print(f"Saved model to {model_path}")

        comparison_rows.append({**cv_summary, **test_summary})

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_csv = RESULTS_DIR / "person3_model_comparison.csv"
    comparison_df.to_csv(comparison_csv, index=False)
    (RESULTS_DIR / "person3_model_comparison.json").write_text(
        json.dumps(comparison_rows, indent=2), encoding="utf-8"
    )
    print(f"\nWrote comparison table to {comparison_csv}")
    print(comparison_df.to_string(index=False))


if __name__ == "__main__":
    main()
