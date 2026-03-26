"""
Evaluation — Profit-based scoring for AHS autoresearch

Two profit metrics:
1. Price profit: simulated gain from buying undervalued properties
2. Insurance profit: gain from identifying under-priced insurance policies

Combined into a single `combined_annual_profit` metric.
"""

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


def compute_regression_metrics(y_true, y_pred):
    """Standard regression metrics."""
    return {
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
        "mape": np.mean(np.abs((y_true - y_pred) / np.clip(y_true, 1, None))) * 100,
    }


def compute_price_profit(y_true, y_pred, config):
    """
    Simulate profit from buying undervalued properties.

    Strategy: buy when model predicts value > actual price by threshold.
    Profit = predicted_value - actual_price - transaction_costs
    """
    scoring = config["scoring"]["price_profit"]
    threshold = scoring["buy_threshold_pct"]
    tx_cost_pct = scoring["transaction_cost_pct"]
    max_invest = scoring["max_investment"]

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    # Identify "buy" signals: predicted value exceeds actual by threshold
    gap_pct = (y_pred - y_true) / np.clip(y_true, 1, None)
    buy_mask = gap_pct > abs(threshold)

    if buy_mask.sum() == 0:
        return {"total_profit": 0.0, "num_trades": 0, "avg_profit_per_trade": 0.0}

    actual_bought = y_true[buy_mask]
    predicted_val = y_pred[buy_mask]

    # Cap investment
    actual_bought = np.clip(actual_bought, 0, max_invest)

    # Profit per trade: predicted upside minus transaction costs
    gross_profit = predicted_val - actual_bought
    tx_costs = actual_bought * tx_cost_pct
    net_profit = gross_profit - tx_costs

    total = float(np.sum(net_profit))
    n_trades = int(buy_mask.sum())

    return {
        "total_profit": total,
        "num_trades": n_trades,
        "avg_profit_per_trade": total / n_trades if n_trades > 0 else 0.0,
    }


def compute_insurance_profit(y_true, y_pred, config):
    """
    Simulate profit from identifying under-priced insurance policies.

    Strategy: flag policies where predicted premium >> actual premium.
    Profit = fraction of the gap captured as adjustment revenue.
    """
    scoring = config["scoring"]["insurance_profit"]
    threshold = scoring["underprice_threshold_pct"]
    capture_pct = scoring["adjustment_capture_pct"]
    min_gap = scoring["min_premium_gap"]

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    # Identify under-priced policies
    gap = y_pred - y_true
    gap_pct = gap / np.clip(y_true, 1, None)
    flag_mask = (gap_pct > threshold) & (gap > min_gap)

    if flag_mask.sum() == 0:
        return {"total_profit": 0.0, "num_flagged": 0, "avg_profit_per_flag": 0.0}

    flagged_gaps = gap[flag_mask]
    captured = flagged_gaps * capture_pct
    total = float(np.sum(captured))
    n_flagged = int(flag_mask.sum())

    return {
        "total_profit": total,
        "num_flagged": n_flagged,
        "avg_profit_per_flag": total / n_flagged if n_flagged > 0 else 0.0,
    }


def compute_combined_profit(price_profit_result, insurance_profit_result, config):
    """Weighted combination of price and insurance profits."""
    weights = config["scoring"]["weights"]
    w_price = weights.get("price_profit", 0.6)
    w_insur = weights.get("insurance_profit", 0.4)

    combined = (
        w_price * price_profit_result["total_profit"]
        + w_insur * insurance_profit_result["total_profit"]
    )
    return combined
