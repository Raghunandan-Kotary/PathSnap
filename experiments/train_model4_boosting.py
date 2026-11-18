"""
Model 4 (Person 2): Histogram-based Gradient Boosting / LightGBM Classifier.
Algorithm Family: Boosting
"""
from sklearn.ensemble import HistGradientBoostingClassifier

def create_model_4(learning_rate: float = 0.05, max_iter: int = 150):
    return HistGradientBoostingClassifier(
        learning_rate=learning_rate,
        max_iter=max_iter,
        class_weight="balanced",
        random_state=42
    )

if __name__ == "__main__":
    print("Model 4 Factory: Gradient Boosting (Boosting Family)")
