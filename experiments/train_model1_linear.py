"""
Model 1 (Person 1): Logistic Regression with Standard Scaling.
Algorithm Family: Linear Models
"""
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from experiments.harness import ModelEvaluationHarness

def create_model_1():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000))
    ])

if __name__ == "__main__":
    print("Model 1 Factory: Logistic Regression (Linear Family)")
