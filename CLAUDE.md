# AHS Autoresearch — Claude Code Instructions

This repo runs **autonomous ML experiments** on American Housing Survey (AHS) data.
Claude Code auto-loads this file on every session.

## Goal
Maximize `combined_annual_profit` — a business metric combining price-prediction
profit and insurance under-pricing detection profit.

## Key files
| File | Purpose |
|------|---------|
| `config/experiment.yaml` | All experiment settings — models, features, thresholds |
| `src/pipeline_runner.py` | Main entry point — reads YAML, orchestrates experiment |
| `src/data_loader.py` | Codebook-informed preprocessing (binary 1/2→1/0, -6→NaN, BUILT midpoints) |
| `src/evaluation.py` | Profit-based scorers for price and insurance tasks |
| `src/model_factory.py` | Model registry — XGBoost, LightGBM, RandomForest, Ridge |
| `program.md` | 6-phase autonomous research plan |
| `run_loop.sh` | Overnight loop runner |
| `data/raw/` | Excel data files (ahs_price_sample.xlsx, ahs_insurance_sample.xlsx) |
| `experiments/results.tsv` | Cumulative experiment log |

## Experiment loop
1. Edit `config/experiment.yaml` (or `src/` code) with a new idea
2. Run `python src/pipeline_runner.py`
3. Check the `METRIC combined_annual_profit=...` output
4. If improved → commit & keep. If worse → revert.
5. Log to `experiments/results.tsv`
6. Repeat indefinitely.

## Rules
- **Never stop** to ask the human. Run autonomously until interrupted.
- The YAML config is the primary knob — change features, models, thresholds, hyperparams there.
- The `src/` modules can also be modified for deeper changes (new scorers, feature engineering, etc.)
- Keep changes small and measurable. One idea per experiment.

## AHS data notes
- Binary columns use AHS coding: 1 = Yes, 2 = No → recode to 1/0
- Code -6 means "Not applicable" → treat as NaN
- BUILT is a range code → convert to midpoint year
- Key columns: VALUE, BUYI, AMTI, UNITSF, LOT, ROOMS, BATHS, CELLAR, GARAGE, etc.
