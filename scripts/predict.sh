#!/usr/bin/env bash

WORKSPACE=${WORKSPACE:-"/hkfs/work/workspace/scratch/ih5525-E2/"}
CONFIG="${WORKSPACE}/Hackasaurus-Rex/configs/detr_prot_predict.yml"
SCRIPT="${WORKSPACE}/Hackasaurus-Rex/scripts/predict.py"

export COMMAND="python -u ${SCRIPT} -c ${CONFIG}"}
sbatch launch_job.sbatch
