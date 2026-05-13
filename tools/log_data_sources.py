"""
tools/log_data_sources.py

Builds a TokenizedBytes dataset (no model, no optimizer) and:
1. Logs which dataset each step draws from (via dataset_index)
2. Optionally inspects N samples per dataset from actual batches

Usage:
    # source logging only (fast, no dataloader):
    torchrun --nproc_per_node=4 tools/log_data_sources.py \
        --config-file my_configs/fw2_iv_fundus_1b_const_lr_diag.yaml \
        --start-step 33000 \
        --steps 10000 \
        --output diagnostics/sources.json

    # also inspect decoded batch contents (slower, needs dataloader):
    torchrun --nproc_per_node=4 tools/log_data_sources.py \
        --config-file my_configs/fw2_iv_fundus_1b_const_lr_diag.yaml \
        --start-step 33205 \
        --steps 10 \
        --output diagnostics/sources.json \
        --inspect-batches \
        --samples-per-dataset 5
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

import nanotron.distributed as dist
from nanotron import logging
from nanotron.config import get_config_from_file, NanosetDatasetsArgs
from nanotron.parallel import ParallelContext
from nanotron.random import set_random_seed
from nanotron.serialize.metadata import DataStageMetadata, TrainingMetadata
from nanotron.data.tokenized_bytes import get_tb_datasets, get_tb_dataloader

try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None

logger = logging.get_logger(__name__)


# ── helpers ───────────────────────────────────────────────────────────────────

def find_stage_for_step(config, step):
    stages = sorted(config.data_stages, key=lambda s: s.start_training_step, reverse=True)
    for stage in stages:
        if step >= stage.start_training_step:
            return stage, config.data_stages.index(stage)
    return config.data_stages[0], 0


def build_dummy_metadata(config, stage, stage_idx, start_step, samples_per_step):
    sequence_length   = stage.sequence_length or config.tokens.sequence_length
    steps_in_stage    = start_step - stage.start_training_step
    consumed_in_stage = steps_in_stage * samples_per_step

    data_stages = [
        DataStageMetadata(
            name=s.name,
            start_training_step=s.start_training_step,
            consumed_train_samples=0,
            consumed_tokens_per_dataset_folder={},
            sequence_length=s.sequence_length or sequence_length,
        )
        for s in config.data_stages
    ]
    data_stages[stage_idx].consumed_train_samples = consumed_in_stage

    metadata = TrainingMetadata(
        consumed_train_samples=consumed_in_stage,
        consumed_tokens_total=0,
        last_train_step=start_step,
        last_stage_idx=stage_idx,
        data_stages=data_stages,
    )
    return metadata


def decode_sample(sample, tokenizer, window=100):
    """
    For each document in the packed sequence (separated by EOS tokens),
    return the first and last `window` tokens.
    """
    ids    = sample.tolist() if hasattr(sample, "tolist") else list(sample)
    eos_id = tokenizer.eos_token_id

    # find all eos positions to locate document boundaries
    eos_positions = [-1] + [i for i, t in enumerate(ids) if t == eos_id]

    docs = []
    for i in range(len(eos_positions)):
        doc_start = eos_positions[i] + 1
        doc_end   = eos_positions[i + 1] + 1 if i + 1 < len(eos_positions) else len(ids)
        doc_ids   = ids[doc_start:doc_end]
        if not doc_ids:
            continue
        docs.append({
            "doc_idx" : i,
            "doc_len" : len(doc_ids),
            "start"   : tokenizer.decode(doc_ids[:window],  skip_special_tokens=False),
            "end"     : tokenizer.decode(doc_ids[-window:], skip_special_tokens=False),
        })

    return {
        "num_docs_in_sequence" : len(docs),
        "unique_tokens"        : len(set(ids)),
        "len"                  : len(ids),
        "docs"                 : docs,
    }



# ── main ──────────────────────────────────────────────────────────────────────

def main(config_path, start_step, n_steps, output_path, inspect_batches, samples_per_dataset):

    config = get_config_from_file(config_path)

    # ── parallel context only — no model, no optimizer ────────────────
    parallel_context = ParallelContext(
        tensor_parallel_size=config.parallelism.tp,
        pipeline_parallel_size=config.parallelism.pp,
        data_parallel_size=config.parallelism.dp,
        expert_parallel_size=config.parallelism.expert_parallel_size,
        context_parallel_size=config.parallelism.context_parallel_size,
    )

    tp_rank = dist.get_rank(parallel_context.tp_pg)
    set_random_seed(config.general.seed + tp_rank)

    # ── derive key sizes ──────────────────────────────────────────────
    stage, stage_idx  = find_stage_for_step(config, start_step)
    stage_data        = stage.data
    data_config       = stage_data.dataset
    sequence_length   = stage.sequence_length or config.tokens.sequence_length
    micro_batch_size  = config.tokens.micro_batch_size
    n_micro_batches   = config.tokens.batch_accumulation_per_replica
    global_batch_size = micro_batch_size * n_micro_batches * config.parallelism.dp
    samples_per_step  = global_batch_size
    local_batch_size  = micro_batch_size * n_micro_batches
    dp_rank           = dist.get_rank(parallel_context.dp_pg)
    dp_size           = parallel_context.dp_pg.size()

    assert isinstance(data_config, NanosetDatasetsArgs), (
        f"Expected NanosetDatasetsArgs, got {type(data_config)}."
    )

    print(f"Stage            : '{stage.name}' (starts at step {stage.start_training_step})")
    print(f"Sequence length  : {sequence_length}")
    print(f"Global batch     : {global_batch_size * sequence_length} tokens = {samples_per_step} samples")
    print(f"Micro batch size : {micro_batch_size}")
    print(f"Micro batches/step: {n_micro_batches}")
    print(f"Local batch/rank : {local_batch_size} samples")
    print(f"Start step       : {start_step}")
    print(f"Steps to log     : {n_steps}")
    if inspect_batches:
        print(f"Samples/dataset  : {samples_per_dataset}")

    # ── reconstruct metadata ──────────────────────────────────────────
    metadata           = build_dummy_metadata(config, stage, stage_idx, start_step, samples_per_step)
    current_stage_meta = metadata.data_stages[stage_idx]
    consumed_train_samples_stage       = current_stage_meta.consumed_train_samples
    consumed_tokens_per_dataset_folder = current_stage_meta.consumed_tokens_per_dataset_folder

    last_stages_consumed_tokens_per_dataset_folder = {}
    for s in metadata.data_stages[:stage_idx]:
        for folder_path, tokens in s.consumed_tokens_per_dataset_folder.items():
            last_stages_consumed_tokens_per_dataset_folder[folder_path] = (
                last_stages_consumed_tokens_per_dataset_folder.get(folder_path, 0) + tokens
            )

    print(f"Consumed samples in stage: {consumed_train_samples_stage}")

    # ── tokenizer ─────────────────────────────────────────────────────
    tokenizer_path = config.tokenizer.tokenizer_name_or_path
    tokenizer      = AutoTokenizer.from_pretrained(tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    eos_token_id = tokenizer.eos_token_id

    # ── pp ranks ──────────────────────────────────────────────────────
    pp_size        = parallel_context.pp_pg.size()
    input_pp_rank  = 0
    output_pp_rank = pp_size - 1

    # ── build dataset ─────────────────────────────────────────────────
    print("\nBuilding TokenizedBytes dataset (no model, no GPU)...")
    start_time = time.time()

    train_dataset, data_log = get_tb_datasets(
        config=data_config,
        global_batch_size=samples_per_step,
        sequence_length=sequence_length,
        train_steps=config.tokens.train_steps,
        current_iteration=start_step,
        parallel_context=parallel_context,
        shuffle=data_config.shuffle_files,
        eos_token_id=eos_token_id,
        seed=stage_data.seed,
        consumed_samples=consumed_train_samples_stage,
        consumed_tokens_per_dataset_folder=consumed_tokens_per_dataset_folder,
        last_stages_consumed_tokens_per_dataset_folder=last_stages_consumed_tokens_per_dataset_folder,
    )
    print(f"Dataset built in {time.time() - start_time:.1f}s")

    # ── dataset names and weights from data_log ───────────────────────
    ds_names   = [Path(s.folder_path).name for s in data_log.train_subset.blended_subset]
    ds_weights = [
        round(n / data_log.train_subset.blended_total_num_samples, 4)
        for n in data_log.train_subset.blended_per_subset_samples
    ]
    print(f"Dataset names  : {ds_names}")
    print(f"Dataset weights: {ds_weights}\n")
    print(f"blended_per_subset_samples: {data_log.train_subset.blended_per_subset_samples}\n")

    # ── source index array ────────────────────────────────────────────
    dataset_index = train_dataset.dataset_index  # shape: (total_train_samples,)
    sample_start  = consumed_train_samples_stage

    # ── optionally build dataloader ───────────────────────────────────
    dataloader_iter = None
    if inspect_batches:
        print("Building dataloader for batch inspection...")
        dataloader = get_tb_dataloader(
            dataset=train_dataset,
            sequence_length=sequence_length,
            micro_batch_size=micro_batch_size,
            global_batch_size=samples_per_step,
            num_workers=stage_data.num_loading_workers,
            cfg=data_config,
            consumed_samples=consumed_train_samples_stage,
            num_samples=config.tokens.train_steps * samples_per_step,
            parallel_context=parallel_context,
            input_pp_rank=input_pp_rank,
            output_pp_rank=output_pp_rank,
            dataloader_drop_last=True,
            dataloader_pin_memory=True,
            use_position_ids=False,
            use_doc_masking=getattr(data_config, "use_doc_masking", None),
        )
        dataloader_iter = iter(dataloader)

    # ── main loop ─────────────────────────────────────────────────────
    print(f"Logging sources for steps {start_step} → {start_step + n_steps}...\n")
    records = []

    for step in range(n_steps):
        actual_step = start_step + step
        step_start  = sample_start + step * samples_per_step
        step_end    = step_start + samples_per_step

        if step_end > len(dataset_index):
            print(f"Reached end of dataset_index at step {actual_step}, stopping.")
            break

        # ── source breakdown (full global batch of 128) ───────────────
        step_sources     = dataset_index[step_start:step_end]  # shape: (128,)
        counts           = np.bincount(step_sources, minlength=len(ds_names)).tolist()
        source_idx       = int(np.argmax(counts))
        source_name      = ds_names[source_idx]
        sample_breakdown = {ds_names[i]: counts[i]                                    for i in range(len(ds_names))}
        token_breakdown  = {ds_names[i]: counts[i] * sequence_length                  for i in range(len(ds_names))}
        pct_breakdown    = {ds_names[i]: round(counts[i] / samples_per_step * 100, 2) for i in range(len(ds_names))}

        record = {
            "step"   : actual_step,
            "source" : source_idx,
            "name"   : source_name,
            "samples": sample_breakdown,
            "tokens" : token_breakdown,
            "pct"    : pct_breakdown,
        }

        if step % 200 == 0:
            print(f"  step={actual_step}  dominant={source_name}  pct={pct_breakdown}")

        # ── optional batch inspection — N samples per dataset ─────────
        if dataloader_iter is not None:

            # Consume all local micro-batches that make up this rank's contribution
            # to the current global step.
            all_ids = np.concatenate(
                [next(dataloader_iter)["input_ids"] for _ in range(n_micro_batches)],
                axis=0,
            )  # shape: (local_batch_size, seq_len)

            local_global_indices = []
            for micro_batch_idx in range(n_micro_batches):
                micro_batch_global_start = step_start + micro_batch_idx * (micro_batch_size * dp_size)
                rank_start = micro_batch_global_start + dp_rank * micro_batch_size
                rank_end = rank_start + micro_batch_size
                local_global_indices.extend(range(rank_start, rank_end))

            assert len(local_global_indices) == len(all_ids), (
                f"Mismatch between computed local sample indices ({len(local_global_indices)}) "
                f"and rank-local batch size ({len(all_ids)})"
            )

            local_sources = [int(dataset_index[idx]) for idx in local_global_indices]
            local_source_sample_indices = [int(train_dataset.dataset_sample_index[idx]) for idx in local_global_indices]

            # Group this rank's batch positions by their source dataset.
            ds_to_positions = {i: [] for i in range(len(ds_names))}
            for pos, src in enumerate(local_sources):
                ds_to_positions[int(src)].append(pos)

            print(f"\n{'═'*70}")
            print(f"  step={actual_step}  rank={dp_rank}  total_local_samples={len(all_ids)}")
            print(f"{'═'*70}")

            batch_inspection = {}
            for ds_idx, ds_name in enumerate(ds_names):
                positions = ds_to_positions[ds_idx]
                print(f"\n  ── {ds_name} ({len(positions)} samples in batch) ──")

                if not positions:
                    print("    (no samples from this dataset in this batch)")
                    batch_inspection[ds_name] = []
                    continue

                ds_samples = []
                for pos in positions[:samples_per_dataset]:
                    decoded = decode_sample(all_ids[pos], tokenizer)
                    print(
                        f"\n    batch_pos={pos}  global_sample_idx={local_global_indices[pos]}  "
                        f"docs_in_seq={decoded['num_docs_in_sequence']}  unique_tokens={decoded['unique_tokens']}"
                    )
                    for doc in decoded["docs"]:
                        print(f"\n      doc {doc['doc_idx']}  len={doc['doc_len']}")
                        print(f"      [START] {doc['start']!r}")
                        print(f"      [END]   {doc['end']!r}")
                    ds_samples.append(
                        {
                            "batch_pos": pos,
                            "global_sample_idx": local_global_indices[pos],
                            "source_dataset_idx": local_sources[pos],
                            "source_dataset_name": ds_names[local_sources[pos]],
                            "source_sample_idx": local_source_sample_indices[pos],
                            **decoded,
                        }
                    )

                batch_inspection[ds_name] = ds_samples

            record["batch_inspection"] = batch_inspection

        records.append(record)

    # ── save on rank 0 ────────────────────────────────────────────────
    if dist.get_rank(parallel_context.world_pg) == 0:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump({"ds_names": ds_names, "ds_weights": ds_weights, "records": records}, f, indent=2)
        print(f"\nDone. Saved {len(records)} records → {output_path}")

    dist.barrier(parallel_context.world_pg)


# ── entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file",         required=True)
    parser.add_argument("--start-step",          type=int, default=0)
    parser.add_argument("--steps",               type=int, default=2000)
    parser.add_argument("--output",              default="diagnostics/sources.json")
    parser.add_argument("--inspect-batches",     action="store_true",
                        help="Decode samples per dataset from each batch")
    parser.add_argument("--samples-per-dataset", type=int, default=3,
                        help="Number of samples to decode per dataset per batch (default: 3)")
    args = parser.parse_args()

    main(
        args.config_file,
        args.start_step,
        args.steps,
        args.output,
        args.inspect_batches,
        args.samples_per_dataset,
    )
