"""
Model 5 (Person 2): Multi-Layer Perceptron Classifier.
Algorithm Family: Neural Networks
"""
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def create_model_5(hidden_layer_sizes=(64, 32), max_iter: int = 200):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation="relu",
            max_iter=max_iter,
            random_state=42,
            early_stopping=True
        ))
    ])

if __name__ == "__main__":
    print("Model 5 Factory: Multi-Layer Perceptron (Neural Network Family)")
