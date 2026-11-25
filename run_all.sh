#!/bin/bash
# Full pipeline, written to survive a dropped connection.
#
# Order: new measures -> probe on the 17 -> corrected analysis -> reports.
# Every stage writes incrementally, so an interruption loses at most the
# current stage; relaunching the script resumes where it stopped, because
# measures.py and probe.py skip keys that are already written.
set -u
cd "$(dirname "$0")"
# Override with: PY=/path/to/python ./run_all.sh
PY=${PY:-python3}
STEP() { echo "=== [$(date +%H:%M:%S)] $* ==="; }

STEP "1/7 geometry of the 5 new checkpoints"
$PY -u code/measures.py --tests batch3 --n 200 --out results

STEP "2/7 residual flow across the 9 objectives"
$PY -u code/measures.py --tests flow3 --n 200 --out results

STEP "3/7 RGB control"
$PY -u code/measures.py --tests rgb --n 200 --out results

STEP "4/7 linear probe on the 17 encoders, 3 splits"
$PY -u code/probe.py --n 387

STEP "5/7 final analysis (encoder-level permutation, recipe vs objective)"
$PY -u code/analysis_final.py

STEP "6/7 review items: interval, width control, stability"
$PY -u code/inference.py
$PY -u code/partial.py
$PY -u code/partial_fl.py
$PY -u code/analysis_stability.py

STEP "7/7 held-out encoder test"
$PY -u code/second_collapse.py
$PY -u code/analysis_second.py

STEP "DONE"
echo "Summary in results/NUMBERS.md"
