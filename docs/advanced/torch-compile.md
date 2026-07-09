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

For `rope` with `data.packing.block_cross_document_attention=true`, auto scope downgrades toward
FFN-focused compile to avoid shape-churn recompiles from dynamic pairwise masks.

## Why the RTD wrapper is not compiled

The full RTD wrapper contains dynamic masking, token sampling, and corruption logic that causes graph breaks and instability in compiled graphs. The repo compiles backbone modules only and leaves wrapper orchestration eager.

For native HF DeBERTa backbones, the compile path does not compile the public
optional-heavy `forward` entrypoint directly. It installs a small eager
dispatcher that normalizes public options, then routes into stable compiled
dense/masked fast paths with fixed output contracts. This avoids repeated Dynamo
specialization on `attention_mask is None` versus tensor and on
`output_hidden_states=None|True|False`.

## FlashDeBERTa under compile

Route selection, kernel tuning, and route-level caveats are covered in
[Advanced / FlashDeBERTa attention](flash-attention.md). This section covers only how the flash
paths stay compile-stable.

FlashDeBERTa path counters are debug-only and disabled by default. Normal compiled training does
not mutate Python stats or emit per-call warnings from inside attention forward; use the
benchmark/probe tooling for path visibility instead.

Flash CUDA routes cross opaque `torch.library` custom-op boundaries: fixed and varlen attention,
flash-with-bias attention, ragged doc-block attention, and fused position-bias attention. Dense
bias assembly is private Triton work inside those production paths, so Dynamo sees a stable
attention primitive rather than upstream Python autograd wrappers, config caches, or launch setup.
In particular, the `bias_docblock_specialized` backward stays behind the same op boundary -
specialization changes which kernels launch, not what Dynamo sees.

Two metadata contracts matter for graph stability:

- Ragged doc-block batches keep the 2D keep mask plus fixed-shape segment descriptors sized to
  `B*S`. The opaque doc-block op slices the active segment prefix, repacks those document spans
  into a ragged batch, runs the disentangled varlen kernels, and scatters results back into the
  packed layout. The fixed-shape descriptors keep the compiled `masked_docblock_*` entrypoints
  from recompiling when the number of documents per packed batch changes.
- Dense (unpadded) batches route through a fast path that takes no flash metadata at all;
  dense-route metadata is semantically empty by contract, so nothing dynamic threads through the
  Dynamo guards.

The padded-varlen custom op uses a `B,S,H,D` internal layout and repo-local prefix-pack Triton
kernels. Because repo masks use standard prefix padding, active tokens pack and repad directly
from `seqlens`/`cu_seqlens` instead of generic `nonzero`/`gather`/`index_copy` flows, and the
backward fuses packed `grad_out` construction with `delta` computation. When profiling unpacked
runs, expect the remaining cost to be the varlen Triton backward kernel plus the smaller
prefix pack/unpack kernels, not `aten::index`-style hotspots.
