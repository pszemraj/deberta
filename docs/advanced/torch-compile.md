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

## FlashDeBERTa notes

FlashDeBERTa path counters are debug-only and disabled by default. Normal compiled
training should not mutate Python stats or emit per-call warnings from inside
attention forward. Use benchmark/probe tooling for path visibility instead.

The current varlen FlashDeBERTa unpadding metadata still lives outside compiled
graphs. That is an intentional correctness-first boundary; if varlen+compile is
still slower than expected after removing Python-side recompiles, treat that as
a separate optimization problem.

## Special case: packed doc-block masks

For `rope` with `data.packing.block_cross_document_attention=true`, auto scope downgrades toward FFN-focused compile to avoid shape-churn recompiles from dynamic pairwise masks.

## Compile debugging helpers

- `local-scratch/compile_parity_check.py`
- `tools/compile_drift_probe.py`
- `local-scratch/hf_attention_inductor_repro.py`
