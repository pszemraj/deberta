# FlashDeBERTa attention

FlashDeBERTa provides optional Triton implementations of disentangled attention for the native `hf_deberta_v2` backbone. Hardware support and the shipped tuning scope are described in [GPU support](gpu-support.md).

## Enable

Install the `flash` extra as described in [Installation](../getting-started/installation.md), then set `model.hf.attention_impl: flash` in the training config.

The flash path has these constraints:

- `model.backbone_type` must be `hf_deberta_v2`.
- `train.mixed_precision` must be `bf16`, and the CUDA device must support bf16.
- Hidden and attention-probability dropout must both be explicitly `0.0`; `null` may preserve
  nonzero backbone/checkpoint dropout and is rejected for flash.
- Every materialized generator and discriminator attention head dimension (`hidden_size / num_attention_heads`) must be a positive power of two.
- The materialized native config must use relative attention with positive position buckets and must not include the unsupported P2P attention term.
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

Route selection comes from `route_policies` in the JSON tuning table. Upstream FlashDeBERTa environment-variable routing is not consulted. Capability-scoped doc-block and local-bias behavior is described in [GPU support](gpu-support.md).

Dense `docblock_bias` is faster on measured shapes but saves a quadratic `(B,H,S,S)` bias for
backward. Ragged `docblock` avoids that allocation and is the conservative choice for hardware or
shapes without a measured dense policy.

## Mask and metadata contracts

The ordinary fixed and varlen routes accept only right-padded prefix masks in exact `(B,S)` or
`(B,1,1,S)` form. Non-prefix masks use eager attention for that call; length mismatches raise.
The collator derives and verifies `seq_lengths` from the CPU mask before device transfer. Batch
preparation selects a route from that metadata. Device masks without collator metadata use eager
attention without a tensor-to-Python layout check in each layer.

Packed doc-block batches accept only metadata prepared through the contract described in the [data pipeline](../guides/data-pipeline.md#cross-document-attention-blocking).

Per-call eager fallbacks preserve semantics:

- `output_attentions=true` uses eager attention to return `(B,H,S,S)` probabilities.
- An explicit `relative_pos` tensor is forwarded through eager attention unchanged.
- A doc-block fallback requires an explicit pairwise mask or the original compact `doc_ids` from
  which it can rebuild one; it never degrades to a padding-only mask.

## Configuration and overrides

All route and override fields live under `model.hf.flash.*`. Their exact null, zero, and
exact-length semantics are defined on each field in the
[config reference](../../configs/config_reference.yaml).

An explicit `docblock_bias_seq_len` can bypass table safety bounds and is therefore an opt-in to the dense memory cost. An active `kernel_overrides_path` is loaded and validated with the training config, then merged with the shipped table for route and kernel lookups.

## Tuning table

Defaults live in
[`flashdeberta_kernel_tuning.json`](../../src/deberta/modeling/flashdeberta_kernel_tuning.json):

- `seq_buckets` names sequence-length and density regimes.
- `route_policies` selects routes in the `padding`, `docblock`, and `local_bias` namespaces.
- `kernels` selects Triton launch configurations by route, operation, shape bucket, and compute
  capability.

Kernel route names include `fixed`, `varlen`, `docblock`, `bias`, `dense_bias`, and `bias_docblock_specialized`. Without a matching row, fixed, varlen, doc-block, and generic bias kernels use the repo-owned conservative `(16, 16, 1, 4)` launch tile; the dense-bias builder uses `(64, 64, 2, 4)`; and specialized doc-block backward kernels remain disabled. These deterministic fallbacks ignore the upstream package's environment-variable tuning surface. Use `model.hf.flash.kernel_overrides_path` for auditable per-hardware choices; [GPU support](gpu-support.md) explains why measured overrides are capability-scoped.

### Retuning

1. Sample real batches with
   [`flashdeberta_varlen_tune.py`](../../tools/flashdeberta_varlen_tune.py) or
   [`flashdeberta_bias_tune.py`](../../tools/flashdeberta_bias_tune.py). Both write batch samples,
   summaries, and `best_candidates.json` winner manifests under `local-scratch/benchmarks/flashdeberta/` unless an
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
- [`flashdeberta_parity_test.py`](../../tools/flashdeberta_parity_test.py) checks outputs and
  selected gradients against eager attention.
The runnable Flash training example is [`pretrain_flashdeberta_1024.yaml`](../../configs/flashdeberta/pretrain_flashdeberta_1024.yaml). The tuning tools override its packing and route settings for their own sampled workloads.

Before merging Flash changes, run `bash tools/premerge.sh`. It records the commit and dirty-tree state under `local-scratch/premerge/` while running lint, docstring checks, the full test suite, and CUDA parity.

## Runtime caveats

- The shipped `sm_120` specialized dense doc-block backward atomically accumulates position gradients and is not bitwise reproducible. Comparisons involving that route must use numeric tolerances; the trainer does not run a separate numerical-drift detector.
- Training uses `drop_last=True` because doc-block route choice depends on batch shape; allowing a
  smaller final batch could change routes and trigger recompilation mid-epoch.
- Multi-GPU validation status is tracked in [Distributed training](distributed-training.md#flashdeberta-with-compile).
