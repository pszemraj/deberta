# FlashDeBERTa attention

FlashDeBERTa provides optional Triton implementations of disentangled attention for the native
`hf_deberta_v2` backbone. Shipped route and kernel measurements target `sm_120`; dated benchmark
results are recorded in the [development log](../devlog.md).

## Enable

Install the `flash` extra as described in [Installation](../getting-started/installation.md), then
set `model.hf.attention_impl: flash` or pass `--model.hf.attention_impl flash`.

The flash path has these constraints:

- `model.backbone_type` must be `hf_deberta_v2`.
- Hidden and attention-probability dropout must both be explicitly `0.0`; `null` may preserve
  nonzero backbone/checkpoint dropout and is rejected for flash.
- `max_relative_positions` must cover `max_position_embeddings`. The default `-1` derives a
  compatible span. A shorter explicit span is rejected because eager and flash bucket distant
  positions differently.
- CUDA is required. See [GPU support](gpu-support.md) for supported capabilities and off-table
  behavior.

## Route families

The adapter selects one route per batch:

| Route | Batch regime | Mechanism |
|---|---|---|
| `fixed` | dense batches and padded sequences below `2048` under the shipped policy | fixed-length disentangled flash kernels |
| `varlen` | padded sequences at `2048+` under the shipped policy | prefix-packed variable-length kernels |
| `local_bias` | small, dense training batches where enabled | dense relative bias with flash-with-bias kernels |
| `docblock_bias` | packed doc-block batches where enabled | document masking folded into a dense attention bias |
| `docblock` | packed doc-block batches using the ragged path | document spans repacked and processed independently by varlen kernels |

Every route uses the same canonical signed relative bucket `query_position - key_position` for
both C2P and P2C terms. Route changes must not change the encoder's attention function.

Route selection comes from `route_policies` in the JSON tuning table. Upstream FlashDeBERTa
environment-variable routing is not consulted. The shipped padding policy switches from `fixed`
to `varlen` at `2048`; capability-scoped doc-block and local-bias behavior is described in
[GPU support](gpu-support.md).

Dense `docblock_bias` is faster on measured shapes but saves a quadratic `(B,H,S,S)` bias for
backward. Ragged `docblock` avoids that allocation and is the conservative choice for hardware or
shapes without a measured dense policy.

## Mask and metadata contracts

The ordinary fixed and varlen routes accept only right-padded prefix masks in exact `(B,S)` or
`(B,1,1,S)` form. Non-prefix masks use eager attention for that call; length mismatches raise.
The collator derives and verifies `seq_lengths` from the CPU mask before device transfer. Batch
preparation reconciles any supplied lengths and counters, then carries a static attestation into
the model. Unattested device masks use eager attention without a tensor-to-Python layout check in
each layer.

Packed doc-block batches use a two-stage collator and batch-preparation contract described in the
[Data pipeline](../guides/data-pipeline.md#cross-document-attention-blocking). Dense doc-block
attention consumes a pairwise keep mask. Ragged doc-block attention consumes the two-dimensional
keep mask plus fixed-capacity segment descriptors. Before transfer, validation proves that segments
do not overlap, stay inside their batch rows, exactly cover active positions, match document
boundaries, and agree with cumulative lengths. The doc-block routes do not use
`FlashBatchMeta.seq_lengths`; only the validated metadata object reaches the model.

Per-call eager fallbacks preserve semantics:

- `output_attentions=true` uses eager attention to return `(B,H,S,S)` probabilities.
- An explicit `relative_pos` tensor is forwarded through eager attention unchanged.
- A doc-block fallback requires an explicit pairwise mask or validated segment metadata from which
  it can rebuild one; it never trusts unattested descriptors or degrades to a padding-only mask.

## Configuration and overrides

All route and override fields live under `model.hf.flash.*`. Their exact null, zero, and
exact-length semantics are defined on each field in the
[config reference](../../configs/config-reference.yaml).

An explicit `docblock_bias_seq_len` can bypass table safety bounds and is therefore an opt-in to
the dense memory cost. `kernel_overrides_path` is loaded at route lookup time, so a missing or
malformed file fails when the first flash batch resolves its route rather than during config
parsing.

## Tuning table

Defaults live in
[`flashdeberta_kernel_tuning.json`](../../src/deberta/modeling/flashdeberta_kernel_tuning.json):

- `seq_buckets` names sequence-length and density regimes.
- `route_policies` selects routes in the `padding`, `docblock`, and `local_bias` namespaces.
- `kernels` selects Triton launch configurations by route, operation, shape bucket, and compute
  capability.

Kernel route names include `fixed`, `varlen`, `docblock`, `bias`, `dense_bias`, and
`bias_docblock_specialized`. Without a matching capability row, fixed and varlen kernels can use
upstream selection, repo-local dense-bias kernels use their generic tile, and specialized
doc-block backward kernels remain disabled. [GPU support](gpu-support.md) explains why these
fallbacks are capability-scoped.

### Retuning

1. Sample real batches with
   [`flashdeberta_varlen_tune.py`](../../tools/flashdeberta_varlen_tune.py) or
   [`flashdeberta_bias_tune.py`](../../tools/flashdeberta_bias_tune.py). Both write batch samples,
   summaries, and winning candidates under `local-scratch/benchmarks/flashdeberta/` unless an
   output directory is supplied.
2. Put durable route or kernel winners in an override table and select it with
   `model.hf.flash.kernel_overrides_path`. Scope hardware-specific rows with
   `compute_capability`. Same-name custom sequence buckets replace shipped buckets; new names are
   evaluated first.
3. Promote results into the shipped table only after parity, throughput, and memory checks hold
   across repeated runs.

## Benchmarking and profiling

- [`flashdeberta_microbench.py`](../../tools/flashdeberta_microbench.py) compares eager and flash
  on synthetic dense and padded shapes.
- [`flashdeberta_rtd_profile.py`](../../tools/flashdeberta_rtd_profile.py) profiles complete RTD
  optimizer steps for one config.
- [`flashdeberta_rtd_compare_step.py`](../../tools/flashdeberta_rtd_compare_step.py) compares one
  identical batch and initial state through eager and flash arms.
- [`flashdeberta_parity_test.py`](../../tools/flashdeberta_parity_test.py) checks outputs and
  selected gradients against eager attention.
- [`run_flashdeberta_benchmarks.sh`](../../tools/run_flashdeberta_benchmarks.sh) runs the tracked
  benchmark matrix.

Packed benchmark configs are under [`configs/flashdeberta/`](../../configs/flashdeberta/).

## Runtime caveats

- Dense doc-block position gradients use atomic accumulation, so flash runs are not bitwise
  reproducible. Resume and drift checks must use numeric tolerances.
- Dense flash-with-bias routes trade memory for recomputation by saving `(B,H,S,S)` bias tensors
  for backward. Table bounds keep unmeasured shapes on ragged routes unless explicitly bypassed.
- Python route counters are disabled during `torch.compile`. Host-side batch preparation still
  reports non-prefix eager fallbacks and doc-block route choices once per shape on rank zero.
- Training uses `drop_last=True` because doc-block route choice depends on batch shape; allowing a
  smaller final batch could change routes and trigger recompilation mid-epoch.
- FSDP2, `torch.compile`, and flash work through compatible wrapping boundaries, but the combined
  multi-GPU path does not yet have an end-to-end test. See
  [Distributed training](distributed-training.md).
