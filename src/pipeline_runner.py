"""
AHS Autoresearch Pipeline Runner

Reads config/experiment.yaml, loads AHS data with codebook preprocessing,
trains models, evaluates with profit-based scoring, and logs results.
"""

import sys
import time
import hashlib
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from data_loader import load_config, load_and_preprocess, prepare_features
from model_factory import create_model
from evaluation import (
    compute_regression_metrics,
    compute_price_profit,
    compute_insurance_profit,
    compute_combined_profit,
)

ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = ROOT / "experiments"
RESULTS_FILE = EXPERIMENTS_DIR / "results.tsv"


def run_task(config, task_key, df):
    """Run a single prediction task (price or insurance)."""
    task_cfg = config[task_key]
    model_name = task_cfg["name"]
    target = task_cfg["target"]
    features = task_cfg["features"]
    hyperparams = task_cfg.get("hyperparams", {})

    print(f"\n{'='*50}")
    print(f"  Task: {task_key} | Model: {model_name} | Target: {target}")
    print(f"{'='*50}")

    # Prepare features
    X, y = prepare_features(df, features, target)
    print(f"  Samples: {len(X)} | Features: {X.shape[1]}")

    if len(X) < 10:
        print(f"  ERROR: Too few samples ({len(X)}). Skipping task.")
        return None

    # Split
    test_size = config["data"].get("test_size", 0.2)
    random_state = config["data"].get("random_state", 42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"  Train: {len(X_train)} | Test: {len(X_test)}")

    # Train
    model = create_model(model_name, hyperparams)
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time

    # Predict
    y_pred = model.predict(X_test)

    # Regression metrics
    metrics = compute_regression_metrics(y_test, y_pred)
    print(f"  RMSE:  {metrics['rmse']:>12,.2f}")
    print(f"  MAE:   {metrics['mae']:>12,.2f}")
    print(f"  R2:    {metrics['r2']:>12.4f}")
    print(f"  MAPE:  {metrics['mape']:>12.2f}%")
    print(f"  Time:  {train_time:.2f}s")

    return {
        "model_name": model_name,
        "target": target,
        "metrics": metrics,
        "y_test": y_test,
        "y_pred": y_pred,
        "train_time": train_time,
        "n_train": len(X_train),
        "n_test": len(X_test),
    }


def log_results(run_id, timestamp, config, price_result, insurance_result, combined_profit):
    """Append results to experiments/results.tsv."""
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

    header = "run_id\ttimestamp\tprice_model\tinsurance_model\tprice_rmse\tprice_r2\tinsurance_rmse\tinsurance_r2\tprice_profit\tinsurance_profit\tcombined_annual_profit\tstatus\tdescription\n"

    if not RESULTS_FILE.exists():
        with open(RESULTS_FILE, "w") as f:
            f.write(header)

    price_cfg = config["price_model"]
    insur_cfg = config["insurance_model"]

    p_rmse = price_result["metrics"]["rmse"] if price_result else 0
    p_r2 = price_result["metrics"]["r2"] if price_result else 0
    i_rmse = insurance_result["metrics"]["rmse"] if insurance_result else 0
    i_r2 = insurance_result["metrics"]["r2"] if insurance_result else 0

    price_profit = 0
    insur_profit = 0
    if price_result:
        pp = compute_price_profit(price_result["y_test"], price_result["y_pred"], config)
        price_profit = pp["total_profit"]
    if insurance_result:
        ip = compute_insurance_profit(insurance_result["y_test"], insurance_result["y_pred"], config)
        insur_profit = ip["total_profit"]

    status = "keep" if combined_profit > 0 else "discard"
    desc = f"{price_cfg['name']}+{insur_cfg['name']}"

    row = f"{run_id}\t{timestamp}\t{price_cfg['name']}\t{insur_cfg['name']}\t{p_rmse:.2f}\t{p_r2:.4f}\t{i_rmse:.2f}\t{i_r2:.4f}\t{price_profit:.2f}\t{insur_profit:.2f}\t{combined_profit:.2f}\t{status}\t{desc}\n"

    with open(RESULTS_FILE, "a") as f:
        f.write(row)

    print(f"\n  Results logged to {RESULTS_FILE}")


def main():
    config_path = ROOT / "config" / "experiment.yaml"
    if not config_path.exists():
        print(f"ERROR: Config not found: {config_path}")
        sys.exit(1)

    config = load_config(str(config_path))

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    run_id = hashlib.md5(timestamp.encode()).hexdigest()[:8]

    print("=" * 60)
    print("  AHS Autoresearch Pipeline")
    print(f"  Run ID: {run_id}")
    print(f"  Started: {timestamp}")
    print("=" * 60)

    # Load data
    print("\nLoading data...")
    df_price, df_insurance = load_and_preprocess(config)

    # Run price prediction
    price_result = run_task(config, "price_model", df_price)

    # Run insurance prediction
    insurance_result = run_task(config, "insurance_model", df_insurance)

    # Compute profits
    print(f"\n{'='*50}")
    print("  PROFIT SCORING")
    print(f"{'='*50}")

    price_profit_result = {"total_profit": 0.0, "num_trades": 0}
    insurance_profit_result = {"total_profit": 0.0, "num_flagged": 0}

    if price_result:
        price_profit_result = compute_price_profit(
            price_result["y_test"], price_result["y_pred"], config
        )
        print(f"  Price profit:     ${price_profit_result['total_profit']:>12,.2f}  ({price_profit_result['num_trades']} trades)")

    if insurance_result:
        insurance_profit_result = compute_insurance_profit(
            insurance_result["y_test"], insurance_result["y_pred"], config
        )
        print(f"  Insurance profit: ${insurance_profit_result['total_profit']:>12,.2f}  ({insurance_profit_result['num_flagged']} flagged)")

    combined = compute_combined_profit(price_profit_result, insurance_profit_result, config)

    print(f"\n  METRIC combined_annual_profit={combined:.2f}")
    print()

    # Log
    log_results(run_id, timestamp, config, price_result, insurance_result, combined)

    print("\n" + "=" * 60)
    print("  Pipeline complete!")
    print("=" * 60)

    return combined


if __name__ == "__main__":
    main()
