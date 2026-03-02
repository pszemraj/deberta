# Developer Notes

## Pre-release Compatibility Policy

Until this repo has a stable release, backward compatibility is not required. Favor correctness and simplicity over preserving old checkpoints or configs.

## Compatibility-Breaking Runtime Checks

- Resume/export validate `run_metadata.json` schema version (`RUN_CONFIG_SCHEMA_VERSION`) and fail on unknown versions.
- Resume source snapshots (`model_config.json`, `data_config.json`) are parsed with strict key validation; unknown legacy keys fail fast.
- Resume compares effective model/data config values and rejects incompatible runs.
- Export and resume both normalize compile wrapper key segments (`._orig_mod`) when loading checkpoint weights.

## Known Limitations

1. Packed-streaming resume is not bit-identical.

`PackedStreamingDataset` does not checkpoint packer buffer internals, so resume+replay can diverge slightly at the boundary.

2. Non-finite skip streak counters are not checkpointed.

`lr_mult` is checkpointed, but `nonfinite_skip_streak` / `nonfinite_skip_total` reset on resume.

3. `train_config.json` is not used as a strict resume compatibility gate.

Model/data config mismatches fail fast; train config drift is allowed by design.

## DeBERTa-v2 Parity Gap Closures

- Native HF DeBERTa-v2 attention now supports `pos_att_type=p2p` with matching scale-factor behavior.
- Relative-position values are clamped to `[-max_position+1, max_position-1]` before log bucketing.

## Objective Notes

- RTD parity configs intentionally use `train.mlm_max_ngram=1` (token-level masking).
- Whole-word/n-gram RTD corruption (`mlm_max_ngram > 1`) is tracked as follow-up objective work.

## Follow-up Work

Use [roadmap](../roadmap.md) for pending engineering tasks and migration items.

RoPE Hugging Face custom-model integration planning lives in
[rope_hf_modular_plan.md](rope_hf_modular_plan.md).
