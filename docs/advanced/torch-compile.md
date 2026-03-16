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
kernels and backward passes. The current routing contract is:

- dense batches use fixed flash
- padded `1024` batches use fixed flash with per-example `seq_lengths`
- longer padded batches (`2048+` by default) use varlen flash when the backend
  package exposes the required low-level primitives

That split is deliberate. On the repo's measured unpacked `1024` RTD regime,
the compile-clean fixed path outperformed the varlen backward kernels, while
the varlen path pulled back ahead again for longer padded contexts.
Set `FLASHDEBERTA_VARLEN_MIN_SEQ_LEN=1024` if you need to force the older
"all padded batches go varlen" policy for debugging or machine-specific
comparisons.

Dense packed `1024` has one additional fast path: for small-batch non-causal
training where DeBERTa relative terms are present, the adapter can materialize
the dense relative bias matrix and route through FlashDeBERTa's flash-with-bias
kernels. That path is opaque to Dynamo in the same way as the fixed and varlen
custom ops, so it behaves like a normal compiled attention primitive instead of
tracing through Python-side launcher code.

The padded-varlen custom op now uses a `B,S,H,D` internal layout and repo-local
prefix-pack Triton kernels. Because repo masks use standard prefix padding,
active tokens can be packed and repadded directly from `seqlens/cu_seqlens`
instead of generic `nonzero`/`gather`/`index_copy` flows, and q/k/v or paired
positional tensors can share those pack/unpack launches. When profiling
unpacked runs, expect the remaining costs to be the varlen Triton backward
kernel plus the smaller repo-local prefix pack/unpack kernels, not the older
`aten::index` or generic `gather`/`index_copy` hotspots. The backward path also
fuses padded `grad_out` packing with `delta` construction, so the old
`prefix-pack grad -> _bwd_preprocess_varlen` boundary is no longer a separate
hot path in repo code.

## Special case: packed doc-block masks

For `rope` with `data.packing.block_cross_document_attention=true`, auto scope downgrades toward FFN-focused compile to avoid shape-churn recompiles from dynamic pairwise masks.

## Compile debugging helpers

- `local-scratch/compile_parity_check.py`
- `tools/compile_drift_probe.py`
- `local-scratch/hf_attention_inductor_repro.py`
