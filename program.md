# AHS Housing ML AutoResearch Program

## Objective
Maximize combined annual profit from two models on the 2011 AHS housing survey:
1. **Regression**: predict Ln(VALUE) (house prices). Offer k% of smearing-corrected predicted value.
2. **Classification**: predict BUYI (insurance purchase). Optimize threshold for profit.

- **PRIMARY METRIC**: `combined_annual_profit` (printed by `src/pipeline_runner.py`)
- Higher is better. Extract from stdout: `grep '^METRIC combined_annual_profit=' run.log`
- Scale: 200,000 properties per year (5% of 4M US home sales)

## Setup (Run Once at Start of Session)
1. Read ALL files in `src/` to understand the pipeline — do NOT modify them
2. Read `config/experiment.yaml` for current configuration
3. Read `experiments/history.jsonl` and `experiments/results.tsv` for past experiments
4. If no history exists, run baseline: `python src/pipeline_runner.py 2>&1 | tee experiments/logs/baseline.log`
5. Record baseline in results.tsv

## File Permissions
- **MUTABLE**: `config/experiment.yaml` (hyperparameters, model selection, sampling)
- **MUTABLE**: `config/features/engineered.yaml` (add/remove engineered features)
- **READ-ONLY**: `src/*.py` (pipeline code, evaluation, data loading)
- **READ-ONLY**: `data/` (raw AHS data files)
- **DO NOT**: install new packages, modify evaluation code, change data splits, touch src/

## Business Context
- Rockafella's Properties buys and sells 200,000 homes/year
- Current baseline: static formula predicting Ln(VALUE), offer 90% of predicted price
- Insurance division: earn 30% margin on AMTI for true positives, -$500 per false positive, -$2,000 per false negative
- Status quo: no insurance offered (costs $2,000 per potential buyer)

## Constraints
- Each experiment must complete within 5 minutes wall-clock
- random_seed: 42 everywhere for reproducibility
- Prefer simpler configs at equal performance
- Maximum 3 new engineered features per experiment
- Do not use AMTI as a classification feature (target leakage)

## Research Directions (Priority Order)

### Phase 1: Model Selection (experiments 1-8)
Try each model with reasonable defaults for BOTH regression and classification:
- LightGBM, XGBoost, RandomForest, Ridge/LogisticRegression
- For classification: also try scale_pos_weight in {1, 10, 25}
- Identify best model family for each task

### Phase 2: Sampling Strategy (experiments 9-14)
For classification only (BUYI is 96.2% imbalanced):
- SMOTE with sampling_strategy in {0.2, 0.3, 0.5}
- No resampling + class_weight='balanced'
- Random undersampling
- Pick best sampling strategy, fix it for later phases

### Phase 3: Feature Engineering (experiments 15-30)
- Log transforms: LOG_UNITSF, LOG_LOT, LOG_ZINC2, LOG_VALUE
- AGE = 2011 - BUILT (with decade midpoint adjustment)
- Ratios: ROOMS_PER_BATH, LOT_PER_SQFT
- Composite: TOTAL_PROBLEMS (sum of EVROD+EROACH+CRACKS+HOLES)
- Composite: HAS_AMENITIES (sum of DISH+AIRSYS+GARAGE+PORCH+DISPL)
- HAS_BASEMENT = CELLAR in {1, 2}
- HOUSING_BURDEN = ZSMHC / (ZINC2/12)
- Drop features with <0.5% importance
- Try interaction terms between top-3 features

### Phase 4: Hyperparameter Tuning (experiments 31-60)
For the best model + features + sampling:
- learning_rate: [0.01, 0.03, 0.05, 0.1, 0.2]
- max_depth: [3, 5, 7, 9, -1]
- n_estimators: [200, 400, 600, 800, 1000]
- subsample: [0.6, 0.7, 0.8, 0.9, 1.0]
- colsample_bytree: [0.6, 0.7, 0.8, 0.9, 1.0]
- reg_alpha: [0, 0.1, 1, 10]
- reg_lambda: [0, 0.1, 1, 10]

### Phase 5: Offer Rate + Threshold Fine-Tuning (experiments 61-75)
- Offer rate k: [0.80, 0.82, 0.85, 0.88, 0.90, 0.92, 0.95]
- Classification threshold: fine search with step=0.005 around current optimum
- Smearing factor recalculation with new model

### Phase 6: Ensembles (experiments 76-100)
- Weighted average of top-2 regression models
- Stacking with Ridge meta-learner
- Blending predictions on validation set

## Anti-Patterns (DO NOT)
- Do not use AMTI as a classification feature (deterministic leakage)
- Do not set max_depth > 15 (overfitting on 50K rows)
- Do not set n_estimators > 3000 (timeout risk)
- If 3 consecutive experiments fail to improve, switch to next research phase
- Do not try deep learning / neural networks
- Do not modify the profit calculation logic

## Experiment Loop
```
REPEAT:
  1. Read experiments/history.jsonl and results.tsv
  2. Decide next experiment (follow Research Directions phases)
  3. Modify config/experiment.yaml
  4. git add config/ && git commit -m "experiment: <description>"
  5. python src/pipeline_runner.py 2>&1 | tee experiments/logs/exp_$(date +%s).log
  6. Extract: grep '^METRIC combined_annual_profit=' <logfile>
  7. IF improved over best → keep, update experiments/best_results.json
     IF equal/worse → git revert HEAD --no-edit
     IF crashed → git revert HEAD --no-edit
  8. Append result to experiments/results.tsv and experiments/history.jsonl
  9. Every 10 experiments: summarize findings so far in experiments/notes.md
  10. GOTO 1
```
