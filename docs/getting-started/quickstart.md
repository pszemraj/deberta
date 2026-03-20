# Quickstart

## 1) Tiny CPU smoke run

Create a tiny local text file and run the smoke config:

```bash
mkdir -p data
printf 'hello world\nthis is a tiny test\n' > data/tiny.txt
deberta train configs/tiny_cpu_smoke.yaml
```

## 2) Small parity-style run on FineWeb-Edu

Use the provided small parity config and shorten the run with dotted overrides:

```bash
deberta train configs/pretrain_hf_deberta_v2_parity_small.yaml \
  --train.max_steps 500 \
  --train.checkpoint.output_dir runs/quickstart_hfv2_small \
  --logging.backend none \
  --logging.wandb.enabled false
```

Distributed launch (single node FSDP2 config):

```bash
accelerate launch --config_file configs/accelerate/fsdp2_hf_deberta_1node.yaml --no_python \
  deberta train configs/pretrain_hf_deberta_v2_parity_small.yaml
```

Optional FlashDeBERTa trial run:

```bash
accelerate launch --config_file configs/accelerate/fsdp2_hf_deberta_1node.yaml \
  tools/train_flashdeberta.py train configs/pretrain_hf_deberta_v2_parity_small.yaml
```

Use the flash wrapper only after installing the optional flash runtime.
Use `tools/flashdeberta_microbench.py` to compare eager vs flash on dense and padded regimes before defaulting flash for a specific config or machine.
For `hf_deberta_v2`, packed doc-block routing is now also supported through the flash wrapper. The repo ships a tracked packed doc-block benchmark config at `configs/custom/pretrain_rtd_hf_deberta_v3pos_smol2stage4_1024_wp32k_v2_docblock.yaml`.
Use `tools/run_flashdeberta_benchmarks.sh` when you want the full current-head matrix in one run, including dense/padded microbench cases plus packed and unpacked training logs under a single persistent `local-scratch/benchmarks/flashdeberta/...` output directory. Override that location with `FLASHDEBERTA_BENCH_OUT_DIR=/your/path` when needed. Set `FLASHDEBERTA_INCLUDE_DOCBLOCK=1` to append the packed doc-block eager/flash comparison, and override the doc-block config path with `FLASHDEBERTA_DOCBLOCK_CONFIG_PATH=/your/config.yaml` if needed.
Use `tools/flashdeberta_varlen_tune.py` when you need real loader-sampled routing or kernel-tuning data for longer padded runs. It samples the actual unpacked dataloader, replays those batches through the native HF DeBERTa backbone under fixed or varlen routing, and writes `summary.tsv`, `batches.jsonl`, and `best_configs.json` under `local-scratch/benchmarks/flashdeberta/...`.
On current `hf_deberta_v2` runs, dense packed `1024` batches may route through the repo-local local-bias flash path automatically. Padded `1024` batches now default to the fixed flash path with per-example `seq_lengths`, while longer padded batches (`2048+` by default) use the varlen path.
Packed doc-block batches now use the measured route split too. Exact packed `1024` batches route through dense flash-with-bias by default, and that short path now uses a repo-local fused dense-bias builder before dispatching to the flash bias kernels. Longer packed doc-block batches use the segment-aware flash custom op. The collator keeps compact `doc_ids`; compile metadata either turns those into a dense pairwise keep mask for the short bias route or expands them into fixed-shape segment descriptors for the longer ragged route. Override the short-sequence bias crossover with `FLASHDEBERTA_DOCBLOCK_BIAS_SEQ_LEN=<len>` or disable it with `FLASHDEBERTA_DOCBLOCK_BIAS_SEQ_LEN=0`.
Dense flash-with-bias now has repo-local tuning hooks too: `FLASHDEBERTA_BIAS_FWD_*` and `FLASHDEBERTA_BIAS_BWD_*` only affect the short packed-docblock bias path, so tuning that route no longer perturbs fixed or varlen kernels. Use `tools/flashdeberta_bias_tune.py` to sweep those overrides against sampled real packed doc-block batches and persist the results under `local-scratch/benchmarks/flashdeberta/`.
The fused dense-bias builder underneath that route also has dedicated overrides: `FLASHDEBERTA_DENSE_BIAS_BLOCK_M`, `FLASHDEBERTA_DENSE_BIAS_BLOCK_N`, `FLASHDEBERTA_DENSE_BIAS_NUM_STAGES`, and `FLASHDEBERTA_DENSE_BIAS_NUM_WARPS`. On the current `sm_120` packed-docblock `1024` path, the promoted builder tile is `64 x 128, stages=2, warps=4`.
The repo also ships dedicated longer-context configs for this workflow:
- `configs/custom/pretrain_rtd_hf_deberta_v3pos_smol2stage4_2048_wp32k_v2.yaml`
- `configs/custom/pretrain_rtd_hf_deberta_v3pos_smol2stage4_4096_wp32k_v2.yaml`
When tuning the padded varlen path, prefer `FLASHDEBERTA_VARLEN_FWD_*` and `FLASHDEBERTA_VARLEN_BWD_*` so experiments do not also perturb the fixed-length flash kernels.
When tuning dense fixed-length flash, prefer `FLASHDEBERTA_FIXED_FWD_*` and `FLASHDEBERTA_FIXED_BWD_*` so experiments do not also perturb the padded-varlen path.

For kernel-level profiling, `tools/flashdeberta_microbench.py --profile-dir local-scratch/profiles/...` writes a Chrome trace plus CPU/CUDA key-average tables for the selected eager or flash attention regime.

## 3) Export discriminator for downstream use

```bash
deberta export runs/quickstart_hfv2_small/checkpoint-500 \
  --what discriminator \
  --output-dir runs/quickstart_hfv2_small/exported_hf
```

## 4) Check run snapshots

Snapshot files and metadata are documented in
[Guides / Configuration](../guides/configuration.md#snapshot-files-and-reproducibility).
