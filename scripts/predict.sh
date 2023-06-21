#!/usr/bin/env bash

WORKSPACE=${WORKSPACE:-"/hkfs/work/workspace/scratch/ih5525-E2/"}
CONFIG="${WORKSPACE}/Hackasaurus-Rex/configs/detr_prot_predict.yml"
SCRIPT="${WORKSPACE}/Hackasaurus-Rex/scripts/predict.py"

export TOMOUNT="${WORKSPACE},"
export COMMAND="python -u ${SCRIPT} -c ${CONFIG}"

echo "Initializing enroot container"
enroot create -n pyxis_torch /hkfs/work/workspace/scratch/ih5525-E2/containers/hackasaurous.sqsh || echo "Container already initialized"

sbatch "${WORKSPACE}/Hackasaurus-Rex/scripts/launch_job.sbatch"
