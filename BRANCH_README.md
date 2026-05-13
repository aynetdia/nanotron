# Nanotron Branch Workflows

This branch contains changes used for Boldt training runs. It is intended to stay close to upstream `huggingface/nanotron` while keeping cluster-specific workflows reproducible for collaborators.

## What Changed

- FlashAttention 2/3 compatibility.
- Resume-from-checkpoint learning-rate patching when YAML LR values differ from the loaded scheduler state.
- Updated dependencies in `pyproject.toml`.
- Slurm entrypoints for training.
- Dataset mixing diagnostics under `tools/`.

## Environment

Create or activate your cluster environment, then install PyTorch for the CUDA runtime available on the node:

```shell
conda create -n nanotron_helma python=3.10 -y
conda activate nanotron_helma
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -e ".[fast-modeling,nanosets]"
```

For FlashAttention 3 on H100 nodes, use CUDA 12.9 or newer and use prebuilt FA wheels from here: https://github.com/mjun0812/flash-attention-prebuild-wheels.

## Training

Export `HF_TOKEN` in the job environment.

```shell
sbatch train_fau_epochs_helma.sh
```

Override GPU count and config file if needed:

```shell
NPROC_PER_NODE=4 CONFIG_FILE=/path/to/config.yaml sbatch --gres=gpu:a100:4 -C a100_80 train_fau_epochs_helma.sh
```

## Convert A Checkpoint

```shell
CKPT=run_name STEP=100000 sbatch eval_ckpt_fau.sh
```

The script reads `tokenizer_name_or_path` from the checkpoint config and writes the checkpoint in HF-compatible format to `${CONVERTED_ROOT}/${CKPT}_${STEP}`.

## Diagnose Data Mixing

Log the dominant source dataset for each step without running model training:

```shell
torchrun --nproc_per_node=4 tools/log_data_sources.py \
  --config-file /path/to/config.yaml \
  --start-step 33000 \
  --steps 10000 \
  --output diagnostics/sources.json
```

Inspect decoded batch samples as well:

```shell
torchrun --nproc_per_node=4 tools/log_data_sources.py \
  --config-file /path/to/config.yaml \
  --start-step 33205 \
  --steps 10 \
  --output diagnostics/sources.json \
  --inspect-batches \
  --samples-per-dataset 5
```
