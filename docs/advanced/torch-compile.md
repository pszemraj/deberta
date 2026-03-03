# torch.compile

## Enable compile

Set:

- `train.torch_compile=true`
- `train.torch_compile_mode` in `default|reduce-overhead|max-autotune|max-autotune-no-cudagraphs`
- `train.torch_compile_backend` in `inductor|aot_eager`

Compile failures raise errors; there is no silent eager fallback.

## Compile scope

`train.torch_compile_scope` supports:

- `auto`
- `backbones`
- `encoder`, `gen_encoder`, `disc_encoder`
- `ffn`, `gen_ffn`, `disc_ffn`

Default behavior (`auto`) compiles heavy backbone modules and keeps unstable wrapper glue in eager mode.

## Why the RTD wrapper is not compiled

The full RTD wrapper contains dynamic masking, token sampling, and corruption logic that causes graph breaks and instability in compiled graphs. The repo compiles backbone modules only and leaves wrapper orchestration eager.

## Special case: packed doc-block masks

For `rope` with `data.block_cross_document_attention=true`, auto scope downgrades toward FFN-focused compile to avoid shape-churn recompiles from dynamic pairwise masks.

## Compile debugging helpers

- `local-scratch/compile_parity_check.py`
- `tools/compile_drift_probe.py`
- `local-scratch/hf_attention_inductor_repro.py`
