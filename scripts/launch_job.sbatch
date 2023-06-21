#!/usr/bin/env bash

# Slurm job configuration
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --job-name=hackasaurous
#SBATCH --partition=accelerated
#SBATCH --reservation=aihero-gpu
#SBATCH --account=hk-project-test-aihero2
#SBATCH --output="/hkfs/work/workspace/scratch/ih5525-E2/slurm_logs/slurm-%j"

ml purge

BASE_DIR="/hkfs/work/workspace_haic/scratch/bk6983-ai_hero_hackathon_shared"
DATA_DIR="/hkfs/work/workspace/scratch/ih5525-energy-train-data"

#export EXT_DATA_PREFIX="/hkfs/home/dataset/datasets/"
#TOMOUNT+="${EXT_DATA_PREFIX},"

TOMOUNT=${TOMOUNT:-""}
TOMOUNT+="${BASE_DIR},${DATA_DIR},"
TOMOUNT+="/scratch,/tmp,"
#TOMOUNT+="/hkfs/work/workspace/scratch/qv2382-dlrt/datasets"
TOMOUNT+='/etc/slurm/task_prolog.hk'
TOMOUNT+=",/hkfs/work/workspace/scratch/qv2382-hackathon/"
TOMOUNT+=",/hkfs/work/workspace/scratch/ih5525-E2/"

export TOMOUNT="${TOMOUNT}"

SRUN_PARAMS=(
  --mpi="pmix"  # TODO: unknown if pmix is there or not!!
  --cpus-per-task=16  # TODO: num
  #--cpu-bind="ldoms"
  # --gpu-bind="closest"
  --label
  --container-name=torch
  --container-writable
  --container-mount-home
  --container-mounts="${TOMOUNT}"
)

export CUDA_VISIBLE_DEVICES="0,1,2,3"
# SCRIPT_DIR="/hkfs/work/workspace/scratch/qv2382-madonna/"
# TODO: set up singularity as well?

export UCX_MEMTYPE_CACHE=0
export NCCL_IB_TIMEOUT=100
export SHARP_COLL_LOG_LEVEL=3
export OMPI_MCA_coll_hcoll_enable=0
export NCCL_SOCKET_IFNAME="ib0"
export NCCL_COLLNET_ENABLE=0

#export CONFIGS="${SCRIPT_DIR}DLRT/configs/"
# export CUDA_VISIBLE_DEVICES="0,1,2,3"

# TODO: make sure to update teh config and the call below this

COMMAND=${COMMAND:-"python -u /hkfs/work/workspace/scratch/qv2382-hackathon/Hackasaurus-Rex/scripts/train.py -c /hkfs/work/workspace/scratch/qv2382-hackathon/Hackasaurus-Rex/configs/detr_prot.yml"}
srun "${SRUN_PARAMS[@]}" bash -c "${COMMAND}"
