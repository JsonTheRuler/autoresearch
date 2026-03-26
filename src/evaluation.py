"""
evaluation.py — Profit-based evaluation metrics for the AHS pipeline.
DO NOT MODIFY. This is the immutable scoring layer.
"""
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, roc_auc_score


def compute_smearing_factor(y_true, y_pred):
    """Duan (1983) smearing estimator for log-retransformation."""
    residuals = y_true - y_pred
    return float(np.mean(np.exp(residuals)))


def regression_profit(y_true_log, y_pred_log, smearing_factor=1.0,
                      offer_rate=0.90, n_annual=200000):
    """Compute regression profit metrics on holdout."""
    actual = np.exp(y_true_log)
    predicted = np.exp(y_pred_log) * smearing_factor
    offers = offer_rate * predicted
    per_unit = actual - offers

    n = len(y_true_log)
    scale = n_annual / n

    return {
        'r2': float(r2_score(y_true_log, y_pred_log)),
        'rmse_log': float(np.sqrt(mean_squared_error(y_true_log, y_pred_log))),
        'mae_dollars': float(np.abs(actual - predicted).mean()),
        'mdape': float(np.median(np.abs(actual - predicted) / actual) * 100),
        'mean_profit_per_unit': float(per_unit.mean()),
        'profitable_pct': float((per_unit > 0).mean() * 100),
        'neg_margin_pct': float((per_unit < 0).mean() * 100),
        'holdout_profit': float(per_unit.sum()),
        'annual_profit': float(per_unit.mean() * n_annual),
    }


def classification_profit(y_true, y_proba, amti, threshold=0.5, n_annual=200000):
    """Compute classification profit using the insurance profit matrix."""
    pred = (y_proba >= threshold).astype(int)
    y = np.asarray(y_true)

    tp = (pred == 1) & (y == 1)
    fp = (pred == 1) & (y == 0)
    fn = (pred == 0) & (y == 1)
    tn = (pred == 0) & (y == 0)

    amti_clean = amti.fillna(amti[y == 1].median())
    amti_arr = np.asarray(amti_clean)

    tp_profit = float((0.3 * amti_arr[tp] - 500).sum())
    fp_cost = float(-500 * fp.sum())
    fn_cost = float(-2000 * fn.sum())
    total = tp_profit + fp_cost + fn_cost

    n = len(y)
    scale = n_annual / n

    try:
        auc = float(roc_auc_score(y, y_proba))
    except:
        auc = 0.0

    return {
        'auc': auc,
        'tp': int(tp.sum()), 'fp': int(fp.sum()),
        'fn': int(fn.sum()), 'tn': int(tn.sum()),
        'tp_profit': tp_profit, 'fp_cost': fp_cost, 'fn_cost': fn_cost,
        'holdout_profit': total,
        'annual_profit': float(total * scale),
        'threshold': float(threshold),
    }


def sweep_thresholds(y_true, y_proba, amti, t_min=0.02, t_max=0.98, t_step=0.02):
    """Find the profit-maximizing threshold on validation data."""
    best_t, best_profit = 0.5, -1e18
    for t in np.arange(t_min, t_max + t_step/2, t_step):
        result = classification_profit(y_true, y_proba, amti, threshold=t)
        if result['holdout_profit'] > best_profit:
            best_profit = result['holdout_profit']
            best_t = float(t)
    return best_t, best_profit
