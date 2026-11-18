"""
Bayesian Hyperparameter Optimization for PathSnap (Person 2).
Uses Optuna with GroupKFold cross-validation over the 80 training trips.
Saves trial history to tuning/trials/optuna_study.csv.
"""
from pathlib import Path
import optuna
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from experiments.metrics import compute_classification_metrics
from models.loader import FEATURE_COLUMNS
from tuning.search_space import define_search_space

def run_optimization(train_df: pd.DataFrame, n_trials: int = 30):
    gkf = GroupKFold(n_splits=5)
    X = train_df[FEATURE_COLUMNS].values
    y = train_df["label"].values
    groups = train_df["trip_id"].values

    def objective(trial):
        params = define_search_space(trial)
        fold_scores = []
        for train_idx, val_idx in gkf.split(X, y, groups):
            clf = HistGradientBoostingClassifier(
                **params,
                class_weight="balanced",
                random_state=42
            )
            clf.fit(X[train_idx], y[train_idx])
            val_probs = clf.predict_proba(X[val_idx])[:, 1]
            val_preds = (val_probs >= 0.5).astype(int)
            f1 = compute_classification_metrics(y[val_idx], val_preds)["macro_f1"]
            fold_scores.append(f1)
        return float(sum(fold_scores) / len(fold_scores))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    
    # Save trials dataframe
    trials_df = study.trials_dataframe()
    out_dir = Path("tuning/trials")
    out_dir.mkdir(parents=True, exist_ok=True)
    trials_df.to_csv(out_dir / "optuna_study.csv", index=False)
    
    print("Optimization Complete!")
    print(f"Best Trial Value (CV Macro F1): {study.best_value:.4f}")
    print(f"Best Parameters: {study.best_params}")
    return study
