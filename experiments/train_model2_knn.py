"""
Model 2 (Person 1): Distance-Weighted K-Nearest Neighbors.
Algorithm Family: Instance-based
"""
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def create_model_2(n_neighbors: int = 15):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier(n_neighbors=n_neighbors, weights="distance"))
    ])

if __name__ == "__main__":
    print("Model 2 Factory: Distance-Weighted KNN (Instance-based Family)")
