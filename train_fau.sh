#!/bin/bash -l

#SBATCH --job-name=fw2_de_run
#SBATCH --output=logs/res_%j.txt     # output file
#SBATCH --partition=a100         # partition to submit to
#SBATCH --gres=gpu:a100:8 -C a100_80
#SBATCH --time=22:00:00
#SBATCH --nodes=1               # Number of nodes

unset SLURM_EXPORT_ENV

module load python

CONDA_ENV=${CONDA_ENV:-/path/to/conda/env}
CONFIG_FILE=${CONFIG_FILE:-/path/to/config.yaml}

if [ ! -d "$CONDA_ENV" ]; then
    echo "Set CONDA_ENV to your nanotron conda environment path." >&2
    exit 1
fi

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Set CONFIG_FILE to a Nanotron YAML config." >&2
    exit 1
fi

conda activate "$CONDA_ENV"

export HF_HOME=${HF_HOME:-$TMPDIR}

# Set these at submit time if your cluster needs an outbound proxy:
# export http_proxy=http://proxy:80
# export https_proxy=http://proxy:80
#
# Also set HF_TOKEN:
# export HF_TOKEN=TOKEN

torchrun --nproc_per_node="${NPROC_PER_NODE:-4}" run_train.py --config-file "$CONFIG_FILE"
