#!/bin/bash
# Source this to activate the project environment:
#   source /data/priorj/sleep_stage_classification/activate.sh
PROJECT_DIR="/data/priorj/sleep_stage_classification"
module load python/3.12.8
source "${PROJECT_DIR}/venv/bin/activate"
cd "${PROJECT_DIR}"
echo "Environment activated: $(python3 --version)"
