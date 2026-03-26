#!/usr/bin/env bash
# AHS Autoresearch Loop Runner
# Usage: bash run_loop.sh [NUM_RUNS]
#
# Runs the pipeline NUM_RUNS times, logging output.
# Designed for overnight autonomous operation.

set -euo pipefail

NUM_RUNS=${1:-10}
LOG_DIR="experiments"
mkdir -p "$LOG_DIR"

echo "=============================================="
echo "  AHS Autoresearch Loop"
echo "  Runs: $NUM_RUNS"
echo "  Started: $(date)"
echo "=============================================="
echo ""

for i in $(seq 1 "$NUM_RUNS"); do
    echo "--- Run $i / $NUM_RUNS [$(date)] ---"

    RUN_LOG="$LOG_DIR/run_${i}.log"
    python src/pipeline_runner.py > "$RUN_LOG" 2>&1
    EXIT_CODE=$?

    if [ $EXIT_CODE -ne 0 ]; then
        echo "  FAILED (exit code $EXIT_CODE) — see $RUN_LOG"
        tail -5 "$RUN_LOG"
    else
        # Extract the key metric
        METRIC=$(grep "METRIC combined_annual_profit" "$RUN_LOG" || echo "METRIC not found")
        echo "  $METRIC"
    fi

    echo ""
done

echo "=============================================="
echo "  All $NUM_RUNS runs complete."
echo "  Results: cat $LOG_DIR/results.tsv"
echo "  Finished: $(date)"
echo "=============================================="
