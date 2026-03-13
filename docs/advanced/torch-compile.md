# torch.compile

## Enable compile

Set:

- `train.compile.enabled=true`
- `train.compile.mode` in `default|reduce-overhead|max-autotune|max-autotune-no-cudagraphs`
- `train.compile.backend` in `inductor|aot_eager`

Compile failures raise errors; there is no silent eager fallback.

## Compile scope

`train.compile.scope` supports:

- `auto`
- `backbones`
- `encoder`, `gen_encoder`, `disc_encoder`
- `ffn`, `gen_ffn`, `disc_ffn`

Default behavior (`auto`) compiles heavy backbone modules and keeps unstable wrapper glue in eager mode.

## Why the RTD wrapper is not compiled

The full RTD wrapper contains dynamic masking, token sampling, and corruption logic that causes graph breaks and instability in compiled graphs. The repo compiles backbone modules only and leaves wrapper orchestration eager.

For native HF DeBERTa backbones, the compile path does not compile the public
optional-heavy `forward` entrypoint directly. It installs a small eager
dispatcher that normalizes public options, then routes into stable compiled
dense/masked fast paths with fixed output contracts. This avoids repeated Dynamo
specialization on `attention_mask is None` versus tensor and on
`output_hidden_states=None|True|False`.

## FlashDeBERTa notes

FlashDeBERTa path counters are debug-only and disabled by default. Normal compiled
training should not mutate Python stats or emit per-call warnings from inside
attention forward. Use benchmark/probe tooling for path visibility instead.

Compiled FlashDeBERTa fixed and varlen paths now both run through opaque custom
ops on CUDA. That keeps the upstream Python autograd wrappers, config caches,
and Triton launch setup out of the Dynamo trace while still executing the real
kernels and backward passes. The compile contract remains the same: dense
batches use fixed flash, and padded batches use varlen flash when the backend
package exposes the required low-level primitives.

The padded-varlen custom op now uses a `B,S,H,D` internal layout so valid tokens
can be flattened with one `B*S` gather/scatter instead of advanced indexing over
`B,H,S,D` transposes. When profiling unpacked runs, expect the remaining costs
to show up as `aten::gather` / `aten::index_copy` and the varlen Triton backward
kernel, not the older `aten::index` hotspot.

## Special case: packed doc-block masks

For `rope` with `data.packing.block_cross_document_attention=true`, auto scope downgrades toward FFN-focused compile to avoid shape-churn recompiles from dynamic pairwise masks.

## Compile debugging helpers

- `local-scratch/compile_parity_check.py`
- `tools/compile_drift_probe.py`
- `local-scratch/hf_attention_inductor_repro.py`
