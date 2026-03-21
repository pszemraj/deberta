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

Packed doc-block batches on `hf_deberta_v2` use a fourth route family with a
measured split by sequence regime. Exact packed `1024` batches route through a
dense flash-with-bias path by default, because that short packed-docblock case
is better served by the quadratic bias kernel than by the ragged varlen
backward path on current GPUs. Longer packed doc-block batches keep the
segment-aware route. In that path, compile metadata keeps the 2D keep mask plus
fixed-shape segment descriptors sized to `B*S`, and the opaque doc-block custom
op slices the active segment prefix, repacks those document spans into a ragged
batch, runs the existing disentangled varlen flash kernels, and scatters the
results back into the original packed layout. The fixed-shape metadata contract
is important: it keeps the compiled `masked_docblock_*` entrypoints from
recompiling when the number of documents per packed batch changes.
The dense-bias branch now builds its `(B,H,S,S)` additive bias through a
repo-local opaque custom op instead of tracing the earlier
`take_along_dim`/mask-scaling chain in eager Python. That keeps dense-bias
assembly compile-stable and materially reduces the packed-docblock `1024`
builder overhead before the flash bias kernels run.
Its backward path now also saves bucket-range metadata through the custom-op
context and reduces dense bias gradients with contiguous segment reductions
instead of the earlier `scatter_add_` / `scatter_reduce_` heavy fallback.
The builder has its own tuning seam as well:
- `FLASHDEBERTA_DENSE_BIAS_BLOCK_M`
- `FLASHDEBERTA_DENSE_BIAS_BLOCK_N`
- `FLASHDEBERTA_DENSE_BIAS_NUM_STAGES`
- `FLASHDEBERTA_DENSE_BIAS_NUM_WARPS`

Those overrides affect only the repo-local dense-bias assembly op, not the
downstream flash-with-bias attention kernels.
The current measured `sm_120` packed-docblock `1024` default for that builder
is `64 x 128, stages=2, warps=4`.
That dense flash-with-bias route now has its own repo-local tuning seam too:
`FLASHDEBERTA_BIAS_FWD_*`, the generic `FLASHDEBERTA_BIAS_BWD_*` fallback, and
the more specific `FLASHDEBERTA_BIAS_BWD_KV_*` / `FLASHDEBERTA_BIAS_BWD_Q_*`
overrides are resolved inside the opaque bias wrapper before it falls back to
upstream FlashDeBERTa config selection, so packed-docblock kernel tuning stays
isolated from the fixed and varlen routes. The repo now launches the raw bias
backward `KV` and `Q` Triton kernels directly, which makes those two backward
surfaces independently tunable without forking the whole attention wrapper.
For the measured packed-docblock `1024` hot path, the wrapper now goes one step
further: when the run is non-causal, bf16/fp16, `D=64`, and the additive bias
is a full dense `(B,H,1024,1024)` tensor, the backward path dispatches exact-
match repo-local `_bwd_kv_kernel_docblock1024` / `_bwd_q_kernel_docblock1024`
Triton kernels instead of the more generic FlashDeBERTa bias backward launcher.
That specialization is intentionally narrow and stays behind the same opaque
custom op boundary, so Dynamo still sees one stable flash-with-bias primitive.
Outside that exact regime, the wrapper falls back to the generic raw bias
backward kernels and the normal override/tuning path.

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

Current repo-local varlen backward heuristics are measured on real unpacked HF
DeBERTa RTD batches on `sm_120`, not synthetic pad ratios:

- `2048_medium` / `2048_sparse`: `KV=(64,32,2,4)`, `Q=(32,64,2,4)`
- `4096_plus`: `KV=(32,64,2,4)`, `Q=(64,64,3,8)`

Those tuned buckets are enough for the current branch to beat eager end to end
on the provided unpacked `2048` and `4096` configs while staying compile-stable.
If you need to retune for another GPU, use `tools/flashdeberta_varlen_tune.py`
first and prefer the split overrides:

- `FLASHDEBERTA_VARLEN_BWD_KV_*`
- `FLASHDEBERTA_VARLEN_BWD_Q_*`

## Special case: packed doc-block masks

For `hf_deberta_v2`, packed doc-blocking now uses the measured route policy
described above when the FlashDeBERTa runtime patch is enabled:

- exact packed `1024` uses dense flash-with-bias by default
- longer packed doc-block runs stay on the segment-aware flash custom op

Set `FLASHDEBERTA_DOCBLOCK_BIAS_SEQ_LEN=0` to disable the dense-bias shortcut,
or point it at a different exact sequence length if another machine bucket
proves a different crossover.

For `rope` with `data.packing.block_cross_document_attention=true`, auto scope
downgrades toward FFN-focused compile to avoid shape-churn recompiles from
dynamic pairwise masks.

## Compile debugging helpers

- `local-scratch/compile_parity_check.py`
- `tools/compile_drift_probe.py`
- `local-scratch/hf_attention_inductor_repro.py`
