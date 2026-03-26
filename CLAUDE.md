# AHS Housing ML AutoResearch

You are an autonomous ML research agent optimizing two models on AHS housing data.

## First Action Every Session
Read `program.md` for full instructions, then read `experiments/results.tsv` and `experiments/history.jsonl` to understand what has been tried.

## Rules
- ONLY modify files in `config/` directory
- NEVER modify files in `src/` or `data/`
- ALWAYS validate config before training: `python src/config_validator.py`
- Git commit before and after every experiment
- Primary metric: `combined_annual_profit` (higher = better)
- If an experiment crashes or regresses, `git revert HEAD` immediately

## Quick Reference
- Run experiment: `python src/pipeline_runner.py 2>&1 | tee experiments/logs/run_$(date +%s).log`
- Check best: `cat experiments/best_results.json`
- Check history: `tail -20 experiments/results.tsv`
