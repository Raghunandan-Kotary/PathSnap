"""
Model 3 (Person 3): Random Forest Ensemble Classifier.
Algorithm Family: Tree Ensemble
"""
from sklearn.ensemble import RandomForestClassifier

def create_model_3(n_estimators: int = 100, max_depth: int = 12):
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1
    )

if __name__ == "__main__":
    print("Model 3 Factory: Random Forest (Ensemble Family)")
