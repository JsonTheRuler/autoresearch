"""
model_factory.py — Creates sklearn/xgb/lgbm model instances from YAML config.
DO NOT MODIFY.
"""
from sklearn.linear_model import Ridge, Lasso, LogisticRegression
from sklearn.ensemble import (RandomForestRegressor, RandomForestClassifier,
                              GradientBoostingRegressor, GradientBoostingClassifier)
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier

REGISTRY = {
    'ridge': Ridge, 'lasso': Lasso,
    'randomforest': RandomForestRegressor, 'rf': RandomForestRegressor,
    'gradientboosting': GradientBoostingRegressor, 'gbr': GradientBoostingRegressor,
    'xgboost': XGBRegressor, 'xgb': XGBRegressor,
    'lightgbm': LGBMRegressor, 'lgbm': LGBMRegressor,
    # Classification variants
    'logistic': LogisticRegression, 'lr': LogisticRegression,
    'randomforest_clf': RandomForestClassifier, 'rf_clf': RandomForestClassifier,
    'gradientboosting_clf': GradientBoostingClassifier, 'gbc': GradientBoostingClassifier,
    'xgboost_clf': XGBClassifier, 'xgb_clf': XGBClassifier,
    'lightgbm_clf': LGBMClassifier, 'lgbm_clf': LGBMClassifier,
}

# Map regression model names to their classifier counterparts
CLF_MAP = {
    'ridge': 'logistic', 'lasso': 'logistic',
    'randomforest': 'randomforest_clf', 'rf': 'rf_clf',
    'gradientboosting': 'gradientboosting_clf', 'gbr': 'gbc',
    'xgboost': 'xgboost_clf', 'xgb': 'xgb_clf',
    'lightgbm': 'lightgbm_clf', 'lgbm': 'lgbm_clf',
}

def create_model(model_cfg):
    """Create a model instance from config dict."""
    name = model_cfg['model_type'].lower()
    params = dict(model_cfg.get('hyperparameters', {}))

    # Determine if this is regression or classification based on context
    # If target is BUYI or if model_type explicitly ends with _clf, use classifier
    is_clf = ('target' in model_cfg and model_cfg['target'] == 'BUYI') or name.endswith('_clf')

    if is_clf and name in CLF_MAP:
        name = CLF_MAP[name]

    if name not in REGISTRY:
        raise ValueError(f"Unknown model type: {name}. Available: {list(REGISTRY.keys())}")

    model_class = REGISTRY[name]
    return model_class(**params)
