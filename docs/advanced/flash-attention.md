# FlashDeBERTa attention

FlashDeBERTa provides optional Triton implementations of disentangled attention for the native `hf_deberta_v2` backbone. Hardware support and the shipped tuning scope are described in [GPU support](gpu-support.md).

## Enable

Install the `flash` extra as described in [Installation](../getting-started/installation.md), then set `model.hf.attention_impl: flash` in the training config.

The flash path has these constraints:

- `model.backbone_type` must be `hf_deberta_v2`.
- `train.mixed_precision` must be `bf16`, and the CUDA device must support bf16.
- Hidden and attention-probability dropout must both be explicitly `0.0`; `null` may preserve
  nonzero backbone/checkpoint dropout and is rejected for flash.
- Every materialized generator and discriminator attention head dimension (`hidden_size / num_attention_heads`) must be a power of two and at least `16`.
- The materialized native config must use relative attention with at least `8` position buckets and the supported C2P, P2C, or combined positional-attention contract. Smaller bucket counts do not preserve eager bucket indices in every fused Triton route.
- `max_relative_positions` must cover `max_position_embeddings`. The default `-1` derives the span from `max_position_embeddings`. A shorter explicit span is rejected because eager and flash bucket distant positions differently.
- CUDA is required. See [GPU support](gpu-support.md) for supported capabilities and off-table
  behavior.

Training and `--dry-run` preflight probe the optional runtime before output or dataset setup. Missing `.[flash]` dependencies or incompatible core low-level imports raise an actionable error instead of silently running eager attention.

For Flash-configured runs, `run_metadata.json` records the configured attention policy and the `flashdeberta` version used by the latest successfully validated training or resume invocation. A resume refreshes that version in its output run metadata; resuming into a new output directory leaves the source run's provenance unchanged. Routing and supported eager fallbacks remain per-call decisions, so the metadata does not claim that every call used one route.

## Route families

The adapter selects one route per batch:

| Route | Batch regime | Mechanism |
|---|---|---|
| `fixed` | dense batches and padded sequences below `2048` under the shipped policy | fixed-length disentangled flash kernels |
| `varlen` | padded sequences at `2048+` under the shipped policy | prefix-packed variable-length kernels |
| `local_bias` | small, dense training batches where enabled | dense relative bias with flash-with-bias kernels |
| `docblock_bias` | benchmark/parity coverage only | document masking folded into a dense attention bias |
| `docblock` | packed doc-block batches using the ragged path | document spans repacked and processed independently by varlen kernels |

Every route uses the same canonical signed relative bucket `query_position - key_position` for
both C2P and P2C terms. Route changes must not change the encoder's attention function.

Route selection comes from `route_policies` in the JSON tuning table. Upstream FlashDeBERTa environment-variable routing is not consulted. Capability-scoped local-bias behavior is described in [GPU support](gpu-support.md).

Normal packed training always uses the ragged `docblock` route, which avoids materializing a quadratic `(B,H,S,S)` bias on every live attention call. The dense `docblock_bias` route is retained for benchmark and parity coverage only; it cannot be selected safely from sequence and batch dimensions alone.

## Mask and metadata contracts

The ordinary fixed and varlen routes accept only right-padded prefix masks in exact `(B,S)` or
`(B,1,1,S)` form. Non-prefix masks use eager attention for that call; length mismatches raise.
The collator derives and verifies `seq_lengths` from the CPU mask before device transfer. Batch
preparation selects a route from that metadata. Device masks without collator metadata use eager
attention without a tensor-to-Python layout check in each layer.

Packed doc-block batches accept only metadata prepared through the contract described in the [data pipeline](../guides/data-pipeline.md#cross-document-attention-blocking).

Per-call eager fallbacks preserve semantics:

- `output_attentions=true` uses eager attention to return `(B,H,S,S)` probabilities.
- An explicit shared `relative_pos` map in exact `(Q,K)`, `(1,Q,K)`, or `(1,1,Q,K)` form is forwarded through eager attention. Batch/head-specific maps and query/key length mismatches raise.
- A doc-block fallback requires an explicit pairwise mask or the original compact `doc_ids` from
  which it can rebuild one; it never degrades to a padding-only mask.

## Configuration and overrides

The optional override path lives under `model.hf.flash.kernel_overrides_path`; its exact semantics are defined in the [config reference](../../configs/config_reference.yaml). The selected table is loaded and validated with the training config, then snapshotted when each model is constructed and carried as model-owned policy through batch routing and forward/backward kernel lookup. Rewriting the source file can affect a subsequently constructed model, but cannot mutate an existing model's policy.

## Tuning table

Defaults live in
[`flashdeberta_kernel_tuning.json`](../../src/deberta/modeling/flashdeberta_kernel_tuning.json):

- `seq_buckets` names sequence-length and density regimes.
- `route_policies` selects routes in the `padding` and `local_bias` namespaces.
- `kernels` selects Triton launch configurations by route, operation, shape bucket, and compute
  capability.

For ordinary padded batches, batch preparation resolves the sequence/density bucket once from active tokens divided by padded capacity. The same bucket controls route selection and every fixed or varlen forward/backward launch, including compiled fixed-capacity packing.

Kernel route names include `fixed`, `varlen`, `docblock`, `bias`, `dense_bias`, and `bias_docblock_specialized`. Without a matching row, fixed, varlen, doc-block, and generic bias kernels use the repo-owned conservative `(16, 16, 1, 4)` launch tile; the dense-bias builder uses `(64, 64, 2, 4)`; and specialized doc-block backward kernels remain disabled. These deterministic fallbacks ignore the upstream package's environment-variable tuning surface. Use `model.hf.flash.kernel_overrides_path` for auditable per-hardware choices; [GPU support](gpu-support.md) explains why measured overrides are capability-scoped.

### Retuning

1. Sample real batches with [`flashdeberta_varlen_tune.py`](../../tools/flashdeberta_varlen_tune.py) or [`flashdeberta_bias_tune.py`](../../tools/flashdeberta_bias_tune.py). Both write batch samples, summaries, and `best_candidates.json` winner manifests under `local-scratch/benchmarks/flashdeberta/` unless an output directory is supplied. Each manifest retains the path to the candidate override table, which already uses the runtime-consumed schema and can be selected directly without transcription.
2. Put durable route or kernel winners in an override table and select it with `model.hf.flash.kernel_overrides_path`. Scope hardware-specific rows with `compute_capability`. Same-name custom sequence buckets replace shipped buckets; new names are evaluated first.
3. Promote results into the shipped table only after parity, throughput, and memory checks hold across repeated runs. Promotion is an explicit review of the winning candidate rows rather than an automated rewrite of package policy.

## Benchmarking and profiling

- [`flashdeberta_microbench.py`](../../tools/flashdeberta_microbench.py) compares eager and flash
  on synthetic dense and padded shapes.
- [`flashdeberta_parity_test.py`](../../tools/flashdeberta_parity_test.py) checks outputs and selected gradients against fp32 eager attention, requiring every Flash result to stay within three times the corresponding eager-bf16 error, subject only to `1e-7` max-error and `1e-8` mean-error numerical-noise floors. Its real-kernel cases cover head dimensions 16, 32, 64, and 128.
The runnable Flash training example is [`pretrain_flashdeberta_1024.yaml`](../../configs/flashdeberta/pretrain_flashdeberta_1024.yaml). The tuning tools override its packing and route settings for their own sampled workloads.

## Verifying Flash changes

`pytest tests/` is the gate. `tests/test_flashdeberta_parity.py` runs the full route matrix above
whenever a CUDA device is visible and skips on CPU-only CI, so no separate pre-merge script is
needed. The `tools/` entry point remains for selecting a single route while debugging:

```bash
python tools/flashdeberta_parity_test.py --case local_bias
```

## Positional-gradient accumulation in fp32

The vendored `flashdeberta==0.0.7` backward kernels scatter the disentangled positional gradients (`grad_pos_key` / `grad_pos_query`) into shared tables with `tl.atomic_add`. The repo wrappers (`flashdeberta_fixed_op.py`, `flashdeberta_varlen_op.py`, and the doc-block route via the shared varlen implementation) allocate those atomic destinations in fp32 and downcast to the model dtype on return. Triton casts the atomic operand to the pointer's element type, so this upgrades the accumulation without modifying the vendored kernels.

This is primarily a numerics fix - it removes the order-sensitive bf16 round-off across the many contributions each table slot receives - but it also turned out to be a large speedup: bf16 global atomics take a slow path on current hardware, while fp32 atomic adds use the native hardware reduction. Measured with `tools/flashdeberta_microbench.py` (forward+backward) on an RTX 5090, idle GPU, and confirmed with the A/B run order reversed (both sides reproduce within 0.4%):

| Config | bf16 destinations | fp32 destinations | delta |
|---|---|---|---|
| fixed route, dense 1024x8 | 89.5 ms / 91.5k tok/s | 55.2 ms / 148.4k tok/s | -38% step time |
| varlen route, 2048x4, pad 0.35 | 100.5 ms / 53.0k active tok/s | 57.4 ms / 92.7k active tok/s | -43% step time |

Peak memory cost is +0.28 GiB on the fixed route (the fp32 tables) and unmeasurable on varlen. The magnitude of the speedup is hardware- and Triton-version-dependent (measured on sm_120, Triton 3.5.1), but the direction holds wherever bf16 global atomics lack a fast native path. The remaining rounding - the kernels downcast `ds * sm_scale` to bf16 before the atomic - is per-contribution, unbiased, and ~2^-9 relative; fixing it requires an upstream kernel change and is parked in [future work](../development/future-work.md#keep-the-vendored-positional-gradient-addend-in-fp32-upstream-kernel-change).

## Runtime caveats

- The benchmark-only `sm_120` specialized dense doc-block backward atomically accumulates position gradients and is not bitwise reproducible. Comparisons involving that route must use numeric tolerances; the trainer does not run a separate numerical-drift detector.
- Training uses `drop_last=True` to keep batch-dependent padding/local-bias routing and compiled shapes stable; a smaller final batch could change routes and trigger recompilation mid-epoch.
- Multi-GPU validation status is tracked in [Distributed training](distributed-training.md#flashdeberta-with-compile).
