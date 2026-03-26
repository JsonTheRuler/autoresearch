"""pipeline_runner.py — Immutable pipeline orchestrator."""
import sys
import yaml
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

from data_loader import load_and_preprocess
from feature_engine import build_features
from model_factory import create_model
from evaluation import (regression_profit, classification_profit,
                        sweep_thresholds, compute_smearing_factor)

def main():
    start = time.time()

    # Load config
    config_path = Path("config/experiment.yaml")
    if not config_path.exists():
        print("ERROR: config/experiment.yaml not found")
        sys.exit(1)

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    print(f"Experiment: {cfg['experiment_name']}")
    print(f"Regression: {cfg['regression']['model_type']}")
    print(f"Classification: {cfg['classification']['model_type']}")
    seed = cfg['random_seed']
    np.random.seed(seed)

    # ========== DATA LOADING ==========
    price_data, insurance_data = load_and_preprocess(cfg)

    X_train_p, X_val_p, X_hold_p = price_data['X_train'], price_data['X_val'], price_data['X_hold']
    y_train_p, y_val_p, y_hold_p = price_data['y_train'], price_data['y_val'], price_data['y_hold']

    X_train_i, X_val_i, X_hold_i = insurance_data['X_train'], insurance_data['X_val'], insurance_data['X_hold']
    y_train_i, y_val_i, y_hold_i = insurance_data['y_train'], insurance_data['y_val'], insurance_data['y_hold']
    amti_val, amti_hold = insurance_data['amti_val'], insurance_data['amti_hold']
    X_resampled, y_resampled = insurance_data.get('X_resampled'), insurance_data.get('y_resampled')

    # Feature names for logging
    feat_names_p = price_data.get('feature_names', [])
    feat_names_i = insurance_data.get('feature_names', [])

    print(f"Price: train={X_train_p.shape}, val={X_val_p.shape}, hold={X_hold_p.shape}")
    print(f"Insurance: train={X_train_i.shape}, val={X_val_i.shape}, hold={X_hold_i.shape}")

    # ========== REGRESSION ==========
    print("\n--- REGRESSION ---")
    reg_model = create_model(cfg['regression'])
    reg_model.fit(X_train_p, y_train_p)

    # Smearing correction
    if cfg['regression'].get('smearing_correction', True):
        smearing = compute_smearing_factor(y_train_p, reg_model.predict(X_train_p))
    else:
        smearing = 1.0
    print(f"Smearing factor: {smearing:.4f}")

    # Holdout predictions
    reg_pred_hold = reg_model.predict(X_hold_p)
    offer_rate = cfg['regression'].get('offer_rate', 0.90)

    reg_metrics = regression_profit(
        y_hold_p, reg_pred_hold,
        smearing_factor=smearing,
        offer_rate=offer_rate,
        n_annual=cfg['scaling']['annual_properties']
    )

    for k, v in reg_metrics.items():
        print(f"  {k}: {v}")

    # Baseline comparison
    baseline_data = price_data.get('baseline_pred')
    if baseline_data is not None:
        bl_metrics = regression_profit(
            y_hold_p, baseline_data,
            smearing_factor=1.0,
            offer_rate=0.90,
            n_annual=cfg['scaling']['annual_properties']
        )
        print(f"  baseline_annual_profit: {bl_metrics['annual_profit']}")

    # Feature importance
    if hasattr(reg_model, 'feature_importances_') and len(feat_names_p) > 0:
        imp = sorted(zip(feat_names_p, reg_model.feature_importances_),
                     key=lambda x: x[1], reverse=True)[:10]
        print(f"  top_features: {[f[0] for f in imp]}")

    # ========== CLASSIFICATION ==========
    print("\n--- CLASSIFICATION ---")
    clf_model = create_model(cfg['classification'])

    # Use resampled data if available, else original
    if X_resampled is not None and cfg['classification']['sampling']['strategy'] != 'none':
        clf_model.fit(X_resampled, y_resampled)
        print(f"  Trained on resampled data ({len(y_resampled)} samples)")
    else:
        clf_model.fit(X_train_i, y_train_i)
        print(f"  Trained on original data ({len(y_train_i)} samples)")

    # Threshold sweep on validation
    val_proba = clf_model.predict_proba(X_val_i)[:, 1]
    thresh_cfg = cfg['classification'].get('threshold_search', {})
    best_threshold, val_profit = sweep_thresholds(
        y_val_i, val_proba, amti_val,
        t_min=thresh_cfg.get('min', 0.02),
        t_max=thresh_cfg.get('max', 0.98),
        t_step=thresh_cfg.get('step', 0.02)
    )
    print(f"  Best threshold (validation): {best_threshold:.3f}")
    print(f"  Validation profit: {val_profit:,.0f}")

    # Holdout evaluation
    hold_proba = clf_model.predict_proba(X_hold_i)[:, 1]
    clf_metrics = classification_profit(
        y_hold_i, hold_proba, amti_hold,
        threshold=best_threshold,
        n_annual=cfg['scaling']['annual_properties']
    )

    for k, v in clf_metrics.items():
        print(f"  {k}: {v}")

    # Status quo and offer-all benchmarks
    sq = -2000 * int((y_hold_i == 1).sum())
    sf = cfg['scaling']['annual_properties'] / len(y_hold_i)
    offer_all = classification_profit(
        y_hold_i, np.ones(len(y_hold_i)), amti_hold,
        threshold=0.5, n_annual=cfg['scaling']['annual_properties']
    )
    print(f"  status_quo_holdout: {sq}")
    print(f"  offer_all_holdout: {offer_all['holdout_profit']}")

    # ========== COMBINED METRICS ==========
    combined = reg_metrics['annual_profit'] + clf_metrics['annual_profit']

    elapsed = time.time() - start
    print(f"\n--- COMBINED ---")
    print(f"METRIC combined_annual_profit={combined:.0f}")
    print(f"METRIC reg_annual_profit={reg_metrics['annual_profit']:.0f}")
    print(f"METRIC clf_annual_profit={clf_metrics['annual_profit']:.0f}")
    print(f"METRIC reg_holdout_r2={reg_metrics['r2']:.4f}")
    print(f"METRIC clf_holdout_auc={clf_metrics['auc']:.4f}")
    print(f"METRIC optimal_threshold={best_threshold:.3f}")
    print(f"METRIC offer_rate={offer_rate:.2f}")
    print(f"METRIC smearing_factor={smearing:.4f}")
    print(f"METRIC elapsed_seconds={elapsed:.1f}")
    print(f"METRIC experiment_name={cfg['experiment_name']}")

    # Log to history
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'experiment_name': cfg['experiment_name'],
        'combined_annual_profit': combined,
        'reg_annual_profit': reg_metrics['annual_profit'],
        'clf_annual_profit': clf_metrics['annual_profit'],
        'reg_model': cfg['regression']['model_type'],
        'clf_model': cfg['classification']['model_type'],
        'reg_r2': reg_metrics['r2'],
        'clf_auc': clf_metrics['auc'],
        'threshold': best_threshold,
        'offer_rate': offer_rate,
        'smearing': smearing,
        'elapsed': elapsed,
        'config': cfg
    }

    history_path = Path("experiments/history.jsonl")
    with open(history_path, 'a') as f:
        f.write(json.dumps(log_entry, default=str) + '\n')

    print(f"\nExperiment completed in {elapsed:.1f}s")

if __name__ == '__main__':
    main()
