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
- longer padded batches (`2048+` in the shipped route table) use varlen flash when the backend
  package exposes the required low-level primitives

That split is deliberate. On the repo's measured unpacked `1024` RTD regime,
the compile-clean fixed path outperformed the varlen backward kernels, while
the varlen path pulled back ahead again for longer padded contexts.
By default this split comes from `src/deberta/modeling/flashdeberta_kernel_tuning.json`.
Set `model.hf.flash.varlen_min_seq_len=1024` if you need to force the older
"all padded batches go varlen" policy for debugging or machine-specific
comparisons.

Dense packed `1024` has one additional fast path: for config-selected small-batch
non-causal training where DeBERTa relative terms are present, the adapter can materialize
the dense relative bias matrix and route through FlashDeBERTa's flash-with-bias
kernels. That path is opaque to Dynamo in the same way as the fixed and varlen
custom ops, so it behaves like a normal compiled attention primitive instead of
tracing through Python-side launcher code.

Packed doc-block batches on `hf_deberta_v2` use a fourth route family. The
JSON route table selects the dense flash-with-bias `docblock_bias` route for
the shipped packed `1024`/`2048`/`4096` lengths; the segment-aware ragged
`docblock` route is the fallback for unlisted shapes and non-`sm_120`
hardware, and can be forced with `model.hf.flash.docblock_bias_seq_len=0`.
In the ragged path, compile metadata keeps the 2D keep mask plus
fixed-shape segment descriptors sized to `B*S`, and the opaque doc-block custom
op slices the active segment prefix, repacks those document spans into a ragged
batch, runs the existing disentangled varlen flash kernels, and scatters the
results back into the original packed layout. The fixed-shape metadata contract
is important: it keeps the compiled `masked_docblock_*` entrypoints from
recompiling when the number of documents per packed batch changes. An earlier
1000-step RTD run showed degenerate convergence on the dense route; that was
traced to a dense output-layout bug (kernel output viewed as `(B,S,H*D)`
without transposing from `(B,H,S,D)`), which is fixed and guarded by a
sentinel regression test. The 2026-07-06 validation campaign (see
`docs/devlog.md`) confirmed dense-route training equivalence with eager at
`1024` and `2048` on top of the speed gates.
The dense-bias branch now builds its `(B,H,S,S)` additive bias through a
repo-local opaque custom op instead of tracing the earlier
`take_along_dim`/mask-scaling chain in eager Python. That keeps dense-bias
assembly compile-stable and materially reduces the packed-docblock `1024`
builder overhead before the flash bias kernels run.
Its backward path now also saves bucket-range metadata through the custom-op
context and reduces dense bias gradients with contiguous segment reductions
instead of the earlier `scatter_add_` / `scatter_reduce_` heavy fallback.
The builder has its own tuning entry in `flashdeberta_kernel_tuning.json`, so
retuning that assembly op does not perturb the downstream flash-with-bias
attention kernels.
The current measured `sm_120` packed-docblock `1024` builder candidate is
`64 x 128, stages=2, warps=4`; it is used whenever the route table selects
dense `docblock_bias`, which is the shipped packed default.
That dense flash-with-bias route now has its own repo-local tuning seam too.
The opaque bias wrapper checks `flashdeberta_kernel_tuning.json` before falling
back to upstream FlashDeBERTa config selection, so packed-docblock kernel tuning
stays isolated from the fixed and varlen routes. The repo now launches the raw
bias backward `KV` and `Q` Triton kernels directly, which makes those two
backward surfaces independently tunable without forking the whole attention
wrapper.
For the packed-docblock dense path at the measured `1024`/`2048`/`4096`
lengths, the wrapper now goes one step further: when the run is non-causal,
bf16/fp16, `D=64`, and the additive bias is a full dense `(B,H,S,S)` tensor, a
matching `bias_docblock_specialized` entry in
`flashdeberta_kernel_tuning.json` enables the repo-local
`_bwd_kv_kernel_docblock1024` / `_bwd_q_kernel_docblock1024` Triton kernels
(historical names; the table gates all three measured square shapes) instead
of the more generic FlashDeBERTa bias backward launcher. That specialization is intentionally
narrow and stays behind the same opaque custom op boundary, so Dynamo still
sees one stable flash-with-bias primitive. Outside that exact regime or table
policy, the wrapper falls back to the generic raw bias backward kernels and the
normal tuning path.

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
first and put durable results in a JSON table selected with
`model.hf.flash.kernel_overrides_path`.

## Special case: packed doc-block masks

For `hf_deberta_v2`, packed doc-blocking uses the JSON route policy described
above when `model.hf.attention_impl=flash` is set:

- measured packed `1024`/`2048`/`4096` buckets use dense flash-with-bias
  `docblock_bias` by default because the direct positional-gradient backward is
  parity-covered and faster than the segment-aware ragged route on the local
  `sm_120` benchmark GPU
- the segment-aware ragged `docblock` route remains available for ablations and
  hardware retuning with `model.hf.flash.docblock_bias_seq_len=0`

The old dense-route learning collapse was traced to an adapter layout bug: the
flash-with-bias kernel returned `(B,H,S,D)` but the adapter viewed it directly as
`(B,S,H*D)`. The adapter now transposes before flattening, and dense doc-block
parity covers `1024`, `2048`, and `4096` forward plus word embedding, relative
embedding, query-projection, and value-projection gradients.

Leave `model.hf.flash.docblock_bias_seq_len` and
`model.hf.flash.local_bias_max_batch_size` unset to use the table. Set
`model.hf.flash.docblock_bias_seq_len=<len>` to force the dense-bias doc-block
route only at that exact sequence length, or set it to `0` to force-disable the
dense-bias doc-block route while keeping flash enabled. Set
`model.hf.flash.local_bias_max_batch_size=0` to disable the small-batch dense
local-bias route without changing the doc-block route. Route policy is
config/table-only; the older FlashDeBERTa route environment fallbacks are not
consulted by the training path or profiling tools.

For `rope` with `data.packing.block_cross_document_attention=true`, auto scope
downgrades toward FFN-focused compile to avoid shape-churn recompiles from
dynamic pairwise masks.

## Compile debugging helpers

- `local-scratch/compile_parity_check.py`
- `tools/compile_drift_probe.py`
- `local-scratch/hf_attention_inductor_repro.py`
