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
  deberta train configs/pretrain_hf_deberta_v2_parity_small.yaml \
  --model.hf.attention_impl flash
```

Use `--model.hf.attention_impl flash` only after installing the optional flash runtime.
Use `tools/flashdeberta_microbench.py` to compare eager vs flash on dense and padded regimes before defaulting flash for a specific config or machine.
For `hf_deberta_v2`, packed doc-block routing is now also supported through the flash config flag. The repo ships a tracked packed doc-block benchmark config at `configs/custom/pretrain_rtd_hf_deberta_v3pos_smol2stage4_1024_wp32k_v2_docblock.yaml`.
Use `tools/run_flashdeberta_benchmarks.sh` when you want the full current-head matrix in one run, including dense/padded microbench cases plus packed and unpacked training logs under a single persistent `local-scratch/benchmarks/flashdeberta/...` output directory. Override that location with `FLASHDEBERTA_BENCH_OUT_DIR=/your/path` when needed. Set `FLASHDEBERTA_INCLUDE_DOCBLOCK=1` to append the packed doc-block eager/flash comparison, and override the doc-block config path with `FLASHDEBERTA_DOCBLOCK_CONFIG_PATH=/your/config.yaml` if needed.
Use `tools/flashdeberta_varlen_tune.py` when you need real loader-sampled routing or kernel-tuning data for longer padded runs. It samples the actual unpacked dataloader, replays those batches through the native HF DeBERTa backbone under fixed or varlen routing, and writes `summary.tsv`, `batches.jsonl`, and `best_configs.json` under `local-scratch/benchmarks/flashdeberta/...`.
On current `hf_deberta_v2` runs, dense packed `1024` batches may route through the repo-local local-bias flash path automatically. Padded `1024` batches now default to the fixed flash path with per-example `seq_lengths`, while longer padded batches (`2048+` in the shipped route table) use the varlen path.
Packed doc-block batches now use the measured route split too. Exact packed `1024` batches route through dense flash-with-bias by default, and that short path now uses a repo-local fused position-bias attention wrapper that avoids saving the full dense bias activation for backward. Longer packed doc-block batches use the segment-aware flash custom op. The collator keeps compact `doc_ids`; compile metadata either turns those into a dense pairwise keep mask for the short bias route or expands them into fixed-shape segment descriptors for the longer ragged route. Leave the flash route fields unset to use the JSON table. Override the short-sequence bias crossover with `--model.hf.flash.docblock_bias_seq_len <len>` or disable it with `--model.hf.flash.docblock_bias_seq_len 0`; set `--model.hf.flash.local_bias_max_batch_size 0` to disable the small-batch dense local-bias route.
Dense flash-with-bias now has repo-local tuning entries too. Use `tools/flashdeberta_bias_tune.py` against sampled real packed doc-block batches, then promote durable results through a JSON table selected by `model.hf.flash.kernel_overrides_path`.
On the current measured packed-docblock `1024` bf16 hot path, that short bias route also enables an exact-match repo-local backward specialization for full dense `(B,H,1024,1024)` bias tensors at `D=64` when the JSON policy table has a matching `bias_docblock_specialized` entry. Every other regime falls back to the generic FlashDeBERTa local-bias path or its table entries.
The fused dense-bias builder underneath that route uses the same table. On the current `sm_120` packed-docblock `1024` path, the promoted builder tile is `64 x 128, stages=2, warps=4`, and the dense-bias backward path now reduces gradients through cached bucket ranges instead of the older scatter-heavy reduction.
The repo also ships dedicated longer-context configs for this workflow:
- `configs/custom/pretrain_rtd_hf_deberta_v3pos_smol2stage4_2048_wp32k_v2.yaml`
- `configs/custom/pretrain_rtd_hf_deberta_v3pos_smol2stage4_4096_wp32k_v2.yaml`
When tuning the padded varlen or dense fixed-length flash paths, prefer table overrides through `model.hf.flash.kernel_overrides_path` so experiments stay reproducible and scoped to the active run config.

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
