#!/usr/bin/env bash
# Run from repo root: cd /data1/julih && bash scripts/run_full_n32_export_then_bootstrap.sh
# 1) Per-subject export for all models (test set n=32)
# 2) Bootstrap 95% CI
# 3) Copy to Results/tables
set -e
cd /data1/julih
python3 scripts/week8_export_per_subject_external.py
python3 scripts/week9/week9_bootstrap_test_set_ci.py --per_subject_dir week8_per_subject_metrics --output_dir week9_stats --B 2000 --seed 42
mkdir -p Results/tables
cp week9_stats/bootstrap_test_set_ci.csv week9_stats/bootstrap_test_set_ci.md Results/tables/
echo "Done. See Results/tables/bootstrap_test_set_ci.csv and .md"
