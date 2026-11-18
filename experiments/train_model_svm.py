"""
Model (Person 3, second model): calibrated Linear SVM.
Algorithm Family: Linear / Margin-based.

The HMM decoder needs well-calibrated probabilities, not just a decision
boundary, so the raw margin from LinearSVC is wrapped in
CalibratedClassifierCV (sigmoid/Platt scaling). Features are standardized
first since SVMs are scale-sensitive, unlike the tree-based models.
"""
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


def create_model_svm(C: float = 1.0, calibration_cv: int = 3):
    base = LinearSVC(C=C, class_weight="balanced", random_state=42, max_iter=5000)
    calibrated = CalibratedClassifierCV(base, method="sigmoid", cv=calibration_cv)
    return make_pipeline(StandardScaler(), calibrated)


if __name__ == "__main__":
    print("Model Factory (Person 3): Calibrated Linear SVM (Linear Family)")
