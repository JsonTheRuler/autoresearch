"""
Model Factory — Supports XGBoost, LightGBM, RandomForest, Ridge

Returns a fitted model given a name and hyperparameters from the YAML config.
"""

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False


def create_model(name, hyperparams=None):
    """Create a regression model by name with given hyperparameters."""
    hyperparams = hyperparams or {}
    name = name.lower().replace("-", "_")

    if name == "xgboost":
        if not HAS_XGBOOST:
            raise ImportError("xgboost not installed. Run: pip install xgboost")
        params = {
            "n_estimators": hyperparams.get("n_estimators", 500),
            "max_depth": hyperparams.get("max_depth", 6),
            "learning_rate": hyperparams.get("learning_rate", 0.05),
            "subsample": hyperparams.get("subsample", 0.8),
            "colsample_bytree": hyperparams.get("colsample_bytree", 0.8),
            "min_child_weight": hyperparams.get("min_child_weight", 5),
            "reg_alpha": hyperparams.get("reg_alpha", 0.1),
            "reg_lambda": hyperparams.get("reg_lambda", 1.0),
            "random_state": hyperparams.get("random_state", 42),
            "n_jobs": -1,
        }
        return xgb.XGBRegressor(**params)

    elif name == "lightgbm":
        if not HAS_LIGHTGBM:
            raise ImportError("lightgbm not installed. Run: pip install lightgbm")
        params = {
            "n_estimators": hyperparams.get("n_estimators", 500),
            "max_depth": hyperparams.get("max_depth", 6),
            "learning_rate": hyperparams.get("learning_rate", 0.05),
            "subsample": hyperparams.get("subsample", 0.8),
            "colsample_bytree": hyperparams.get("colsample_bytree", 0.8),
            "min_child_samples": hyperparams.get("min_child_samples", 10),
            "reg_alpha": hyperparams.get("reg_alpha", 0.1),
            "reg_lambda": hyperparams.get("reg_lambda", 1.0),
            "random_state": hyperparams.get("random_state", 42),
            "verbose": hyperparams.get("verbose", -1),
            "n_jobs": -1,
        }
        return lgb.LGBMRegressor(**params)

    elif name == "random_forest":
        params = {
            "n_estimators": hyperparams.get("n_estimators", 500),
            "max_depth": hyperparams.get("max_depth", None),
            "min_samples_split": hyperparams.get("min_samples_split", 5),
            "min_samples_leaf": hyperparams.get("min_samples_leaf", 2),
            "random_state": hyperparams.get("random_state", 42),
            "n_jobs": -1,
        }
        return RandomForestRegressor(**params)

    elif name == "ridge":
        params = {
            "alpha": hyperparams.get("alpha", 1.0),
        }
        return Ridge(**params)

    else:
        raise ValueError(f"Unknown model: {name}. Choose from: xgboost, lightgbm, random_forest, ridge")
