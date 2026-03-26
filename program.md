# AHS Autoresearch — 6-Phase Research Plan

## Overview
Autonomous ML research on American Housing Survey (AHS) data to maximize
`combined_annual_profit` — a business metric combining property value prediction
profit and insurance under-pricing detection profit.

## Phase 1: Baseline Establishment
**Goal**: Run the default config and record baseline metrics.
1. Run `python src/pipeline_runner.py` with default `config/experiment.yaml`
2. Record the `combined_annual_profit` as the baseline
3. Commit as "baseline" in `experiments/results.tsv`

## Phase 2: Feature Engineering
**Goal**: Improve features fed to the models.
Ideas to try:
- Interaction features: UNITSF × LOT, ROOMS × BATHS
- Price-per-sqft: VALUE / UNITSF
- Age of structure: current_year - BUILT
- Log transforms of skewed features (VALUE, UNITSF, LOT)
- Polynomial features for key predictors
- Binning continuous variables (LOT size categories, age buckets)

## Phase 3: Model Selection & Tuning
**Goal**: Find the best model and hyperparameters for each task.
- Try all 4 models: XGBoost, LightGBM, RandomForest, Ridge
- Grid search key hyperparameters in YAML
- Tune separately for price vs insurance (they may want different models)
- Consider ensemble/stacking if individual models plateau

## Phase 4: Threshold Optimization
**Goal**: Tune the profit scoring thresholds.
- `buy_threshold_pct`: try 0.05, 0.10, 0.15, 0.20
- `transaction_cost_pct`: model realistic vs optimistic scenarios
- `underprice_threshold_pct`: try 0.10, 0.15, 0.20, 0.25
- `adjustment_capture_pct`: sensitivity analysis
- `min_premium_gap`: try 100, 200, 500

## Phase 5: Advanced Techniques
**Goal**: Push the frontier with more sophisticated approaches.
- Target encoding for categorical features
- Feature selection via importance/SHAP
- Quantile regression for confidence intervals
- Custom loss functions aligned with profit metric
- Time-aware splits if YEAR column is meaningful

## Phase 6: Robustness & Validation
**Goal**: Ensure results are stable and not overfit.
- K-fold cross-validation (5 or 10 folds)
- Multiple random seeds
- Check for data leakage
- Feature importance analysis
- Final model documentation

## Experiment Loop Protocol
```
LOOP FOREVER:
1. Pick an idea from the current phase
2. Edit config/experiment.yaml (or src/ code)
3. git commit
4. python src/pipeline_runner.py > run.log 2>&1
5. grep "METRIC combined_annual_profit" run.log
6. If improved → keep commit, log "keep" in results.tsv
7. If worse → git reset, log "discard" in results.tsv
8. If crash → check tail -50 run.log, fix or skip
9. Move to next idea
```

**NEVER STOP** — run autonomously until manually interrupted.
