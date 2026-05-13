#!/bin/bash -l

#SBATCH --job-name=eval_ckpt
#SBATCH --output=logs/res_%j.txt     # output file
#SBATCH --partition=a40         # partition to submit to
#SBATCH --gres=gpu:a40:1
#SBATCH --time=0:15:00
#SBATCH --nodes=1               # Number of nodes

unset SLURM_EXPORT_ENV

module load python

# Uncomment these on FAU if internet access is needed:
# export http_proxy=http://proxy:80
# export https_proxy=http://proxy:80

CONDA_ENV=${CONDA_ENV:-/path/to/conda/env}
MODEL_ROOT=${MODEL_ROOT:-/path/to/checkpoints}
CONVERTED_ROOT=${CONVERTED_ROOT:-/path/to/converted}
CONFIG_CLS=${CONFIG_CLS:-Qwen2Config}
export HF_HOME=${HF_HOME:-$TMPDIR}

if [ -z "$CKPT" ] || [ -z "$STEP" ]; then
	echo "Set CKPT and STEP before submitting, for example: CKPT=run_name STEP=420000 sbatch eval_ckpt_fau.sh" >&2
	exit 1
fi

if [ ! -d "$CONDA_ENV" ]; then
	echo "Set CONDA_ENV to your nanotron conda environment path." >&2
	exit 1
fi

conda activate "$CONDA_ENV"
TOKENIZER=$(grep 'tokenizer_name_or_path:' "$MODEL_ROOT/$CKPT/$STEP/config.yaml" | awk '{print $2}')

torchrun --standalone --nproc_per_node=1 \
	-m examples.llama.convert_nanotron_to_hf \
	--checkpoint_path "$MODEL_ROOT/$CKPT/$STEP" \
	--save_path "$CONVERTED_ROOT/${CKPT}_${STEP}" \
	--tokenizer_name $TOKENIZER \
	--config_cls "$CONFIG_CLS"
conda deactivate
