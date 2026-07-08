# FlashDeBERTa attention

FlashDeBERTa is an optional Triton implementation of DeBERTa's disentangled attention for the
native `hf_deberta_v2` backbone. On the measured benchmark GPU it beats eager end to end on the
repo's RTD configs, including packed doc-block batches (`1.28x`/`1.36x`/`1.33x` tokens/sec at
packed `1024`/`2048`/`4096`, with lower peak memory).

## Enable

1. Install the extra: `pip install -e '.[flash]'` (pins `flashdeberta==0.0.7` plus `triton`).
2. Set `model.hf.attention_impl: flash` in config, or pass `--model.hf.attention_impl flash`.

Constraints:

- valid only with `model.backbone_type=hf_deberta_v2`
- dropout must be disabled (`model.dropout.hidden_prob` and `model.dropout.attention_probs_prob`
  set to `0.0` or null)
- CUDA only. Measured route/kernel defaults ship for `sm_120` (RTX 5090); other GPUs run flash
  with conservative capability-scoped defaults - see [GPU support](gpu-support.md)
- Calls with `output_attentions=true` use eager attention for that call, preserving the HF-style
  `(B,H,S,S)` attention-probability outputs instead of returning flash-only placeholders.
- Calls with an explicit `relative_pos` tensor use eager attention for that call with the tensor
  preserved; the flash kernels only compute the default relative-position map on device.
- Doc-block eager fallbacks fail closed: they reuse an explicit pairwise mask or rebuild one from
  complete segment metadata, and raise otherwise. A doc-block batch is never downgraded to a
  compact 2D padding mask, which would silently allow cross-document attention.
- The `fixed`/`varlen` padding routes compress key-padding masks into per-example right-padded
  prefix lengths, so they only accept exact-shape `(B,S)`/`(B,1,1,S)` contiguous-prefix masks.
  Legal non-prefix masks (holes, left padding) use eager attention for that call, and
  length-mismatched masks raise instead of being silently sliced. `FlashBatchMeta.seq_lengths`
  carries the same right-padding contract; the training metadata prep verifies it before
  publishing lengths and leaves non-prefix batches on eager.

## Route families

The adapter picks one of five routes per batch:

| Route | Batch regime | Mechanism |
|---|---|---|
| `fixed` | dense (unpadded) batches, and padded `1024` with per-example `seq_lengths` | fixed-length disentangled flash kernels |
| `varlen` | longer padded batches (`2048+` in the shipped table) | variable-length kernels over prefix-packed active tokens |
| `local_bias` | plain dense `1024` batches at small batch size (`<= 4`, `sm_120` only in the shipped table) | materializes the dense relative bias and runs flash-with-bias kernels |
| `docblock_bias` | packed doc-block batches at the measured `1024`/`2048`/`4096` lengths on `sm_120` (or wherever a capability row/knob enables it) | dense flash-with-bias with the pairwise document keep-mask folded into the bias; padding query rows are zeroed afterwards to match eager outputs exactly |
| `docblock` | packed doc-block batches elsewhere: other hardware, unlisted shapes, or forced for ablations | segment-aware ragged route: repacks document spans and runs the varlen kernels per document |

Route selection is config/table-only: it reads `route_policies` from the JSON tuning table, and
the upstream FlashDeBERTa environment-variable route fallbacks are never consulted. Policy rows
may be scoped to one GPU class with a `compute_capability` key; an exact `sm_XX` row outranks the
wildcard rows, and the shipped table scopes its aggressive defaults (`docblock_bias`,
`local_bias`) to `sm_120` while the padded `fixed`/`varlen` split applies everywhere. See
[GPU support](gpu-support.md) for the per-hardware picture.

The padded fixed/varlen split at `2048` is deliberate: on the measured unpacked `1024` RTD regime
the compile-clean fixed path beats the varlen backward kernels, while varlen pulls back ahead for
longer padded contexts. The training loop precomputes route hints per batch; when the model
resolves a route itself (no hint), it also rechecks that the installed `flashdeberta` package
exposes the required varlen primitives.

For packed doc-block batches, dense `docblock_bias` is the shipped default at the three measured
lengths on `sm_120` because its direct positional-gradient backward is parity-covered and faster
than the ragged route on the benchmark GPU. Ragged `docblock` remains the correctness-preserving
ablation path and the automatic choice for unlisted packed shapes and other hardware, where the
dense route's speed edge (which depends on the `sm_120`-gated backward specializations) is
unproven and its saved dense bias costs more memory.

## Routing overrides

All flash knobs live under `model.hf.flash.*`; per-key details are in
[Guides / Config Reference](../guides/config-reference.md). The ones that change routing:

- `docblock_bias_seq_len`: `null` uses the table; a positive value allows the dense doc-block
  route only at that exact sequence length; `0` forces the ragged `docblock` route. Only affects
  packed doc-block batches.
- `local_bias_seq_len` / `local_bias_max_batch_size`: the same null/`0`/exact-length semantics
  for the plain-batch dense local-bias route, independent of the doc-block knobs. On hardware
  without a matching table row, set both to opt in.
- `varlen_min_seq_len`: set to `1024` to force the older "all padded batches go varlen" policy
  for debugging or cross-machine comparison.
- `kernel_overrides_path`: path to a JSON table consulted before the shipped one. Keep override
  tables run-local and record them with checkpoints for reproducibility.

## Kernel tuning table

Defaults ship in `src/deberta/modeling/flashdeberta_kernel_tuning.json` with three sections:

- `seq_buckets`: named shape/density buckets (`1024_exact`, `2048_medium`, `2048_sparse`,
  `2048_plus`, `4096_plus`, `under_2048`)
- `route_policies`: route selection per namespace (`padding`, `docblock`, `local_bias`)
- `kernels`: Triton launch configs keyed by route family, kind (`fwd`/`bwd`/`bwd_kv`/`bwd_q`),
  sequence bucket, and compute capability (currently all `sm_120`)

Kernel route families are `fixed`, `varlen`, `docblock` (ragged backward tiles), `bias` (the
local-bias kernels; note the naming split versus the `local_bias` route-policy namespace),
`dense_bias` (the dense bias-assembly builder op), and `bias_docblock_specialized` (the dedicated
dense doc-block backward, gated to non-causal bf16/fp16 `head_dim=64` full-bias shapes). On
hardware without matching kernel entries, routes still apply and kernels fall back to upstream
FlashDeBERTa config selection; see [GPU support](gpu-support.md) for what that means per GPU
class.

Measured `sm_120` highlights: varlen backward runs `KV=(64,32,2,4)` / `Q=(32,64,2,4)` at
`2048_medium`/`2048_sparse` and `KV=(32,64,2,4)` / `Q=(64,64,3,8)` at `4096_plus`; the dense-bias
builder tile is `64 x 128, stages=2, warps=4` at the packed doc-block lengths.

### Retuning for new hardware or kernel changes

1. Sample real batches and measure candidates:
   - `tools/flashdeberta_varlen_tune.py` samples the actual unpacked dataloader, replays batches
     through the backbone under fixed or varlen routing, and writes `summary.tsv`,
     `batches.jsonl`, and `best_configs.json` under `local-scratch/benchmarks/flashdeberta/...`.
   - `tools/flashdeberta_bias_tune.py` does the same for the dense-bias kernels against sampled
     packed doc-block batches.
2. Promote durable winners into a JSON table and select it with
   `model.hf.flash.kernel_overrides_path`. Scope rows to your GPU with a `compute_capability`
   key; override rows append to the shipped table and exact-capability rows outrank wildcards,
   so one row per bucket is enough.
3. Once results hold up across runs, fold them into the shipped table.

## Benchmarking and profiling tools

- `tools/flashdeberta_microbench.py`: eager vs flash on synthetic dense and padded regimes;
  `--profile-dir` writes a Chrome trace plus CPU/CUDA key-average tables.
- `tools/flashdeberta_rtd_profile.py`: end-to-end RTD optimizer-step profiling for one config
  (`--mode eager|flash`, `--warmup-steps`, `--profile-steps`, `--profile-dir`).
- `tools/flashdeberta_rtd_compare_step.py`: one identical batch and identical initial weights
  through an eager arm and a flash arm, reporting activation/logit deltas.
- `tools/flashdeberta_parity_test.py`: strict numeric parity matrix versus eager (outputs plus
  selected gradients); exits non-zero on failure.
- `tools/run_flashdeberta_benchmarks.sh`: the full matrix in one run under a persistent
  `local-scratch/benchmarks/flashdeberta/...` directory. Override the location with
  `FLASHDEBERTA_BENCH_OUT_DIR`, add the packed doc-block comparison with
  `FLASHDEBERTA_INCLUDE_DOCBLOCK=1`, and point at a different doc-block config with
  `FLASHDEBERTA_DOCBLOCK_CONFIG_PATH`.

Tracked packed doc-block configs for benchmarking and training live at
`configs/custom/pretrain_rtd_hf_deberta_v3pos_smol2stage4_{1024,2048,4096}_wp32k_v2_docblock.yaml`
(plus non-doc-block `_v2` siblings).

## Accepted caveats

- Flash gradients are not bitwise reproducible: the specialized dense doc-block backward
  accumulates positional-score gradients with `tl.atomic_add`. Same-seed flash runs diverge at
  bf16 noise scale (measured final-loss spread about `1.7%` relative over 1000 RTD steps at
  packed `1024`, with same-seed eager-vs-flash deltas inside that spread). Resume/drift tooling
  must not assert bit-exact replay through flash attention.
- The flash-with-bias op saves the dense `(B,H,S,S)` bias tensor for backward, trading memory
  for recompute. Peak memory still measures well below eager at the shipped packed configs, but
  a recompute-in-backward knob is a prerequisite before running contexts longer than `4096` or a
  larger batch-times-length product.
- Route counters and fallback warnings are debug-only and off by default; set
  `FLASHDEBERTA_DEBUG_STATS=1` / `FLASHDEBERTA_WARN_FALLBACKS=1` when diagnosing routing in
  eager runs. Compiled training never mutates the Python-side stats (see
  [Advanced / torch.compile](torch-compile.md)).

## Known follow-ups

- Benchmark and, only if warranted, add the upstream small-batch local-bias path for
  `512 < seq_len < 1024` with very small training batches.
- Extend the measured JSON policy table with more hardware buckets and route cases before
  deleting less-used specialized kernel branches.
- Replace the remaining transient dense-bias / `d_bias` allocations with tile-local
  position-bias kernels once the upstream kernel signatures are owned locally.
- Evaluate recomputing position/bucket tensors in backward versus saving them for longer
  contexts (pairs with the dense-bias recompute knob above).
- Add a bucket-LUT path and deterministic position-gradient kernels before treating the
  dense-bias builder as kernel-owned infrastructure.
- Keep reducing the ragged `docblock` route's varlen backward overhead; it currently trails the
  dense route at the shipped packed lengths.
- Vendor or replace the upstream FlashDeBERTa kernels if this integration becomes a long-lived
  training dependency (currently pinned to `flashdeberta==0.0.7`).

Measurement history and the decisions behind the shipped route policy are recorded in the
[development log](../devlog.md).
