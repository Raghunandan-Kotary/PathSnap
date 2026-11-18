"""
Bayesian optimization hyperparameter search spaces (Optuna distributions).
Defined for the chosen model family (Gradient Boosting).
"""
def define_search_space(trial):
    """Defines search distributions for Gradient Boosting."""
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_iter": trial.suggest_int("max_iter", 50, 250, step=25),
        "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 63),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 50),
        "l2_regularization": trial.suggest_float("l2_regularization", 1e-4, 10.0, log=True)
    }
