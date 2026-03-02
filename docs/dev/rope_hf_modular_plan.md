# RoPE HF Interop Plan (Modular Transformers)

## Goal

Enable exported RoPE checkpoints to load in the Hugging Face ecosystem with `AutoConfig`/`AutoModel` compatibility, while keeping the current export contract (`discriminator` + tokenizer + config) stable during rollout.

## Scope

- Plan-only artifact for this PR.
- No RoPE HF runtime code or export behavior changes in this PR.
- Focus on architecture, packaging, and test strategy for subsequent implementation PRs.

## Baseline Constraints

- RoPE backbone is not `hf_deberta_v2` and must not be represented as such.
- Current stable export path is best-effort discriminator export plus tokenizer files.
- `transformers` modular converter (`utils/modular_model_converter.py`) is a source-generation tool, not a runtime dependency.

## Proposed Architecture

### Runtime targets

1. `DebertaRopeConfig`
- `model_type = "deberta-rope"`
- includes RoPE-specific fields (`rope_theta`, `rotary_pct`, `norm_arch`, `use_absolute_position_embeddings`, etc.)

2. `DebertaRopeModel`
- encoder-only base model class mapped to exported discriminator weights

3. Optional task heads (later)
- `DebertaRopeForMaskedLM`
- `DebertaRopeForSequenceClassification`

### Packaging mode

Use `trust_remote_code`-compatible packaging first:

- export directory includes:
  - `configuration_deberta_rope.py`
  - `modeling_deberta_rope.py`
  - `__init__.py`
  - `config.json` with `auto_map`

This avoids requiring upstream `transformers` merge before users can load exported models.

## Modular Transformers Strategy

Use modular files as the authoring layer, flattened files as release artifacts.

### Authoring

- Create `modular_deberta_rope.py` in a pinned local transformers worktree.
- Inherit from DeBERTa-v2 classes where useful; override only changed behavior.

### Generation

- Run converter:
  - `python utils/modular_model_converter.py deberta_rope`
- Commit/generated output is the flat single-file runtime artifact used by exported models.

### Repository integration

- Store generated runtime files in this repo under a dedicated codegen path.
- Keep generation script + pinned transformers revision documented for deterministic refresh.

## DeBERTa-Specific Implementation Notes

- Attention modifications must target `DisentangledSelfAttention` when building on DeBERTa-v2 internals.
- Re-point class chain explicitly: `Layer -> Encoder -> Model`.
- For DeBERTa attention wrappers, replace at `self.attention.self`, not `self.attention`.
- If deep overrides are not re-pointed, converter output will retain parent behavior.

## Phased Implementation Plan

### Phase 1: HF custom-code load path

- Add `DebertaRopeConfig` + `DebertaRopeModel` runtime files in this repo.
- Extend export to optionally emit code files + `auto_map`.
- Keep default export behavior unchanged unless explicit flag enables code emission.

### Phase 2: Modular authoring pipeline

- Add generator workflow doc + helper script to rebuild flat modeling files from modular source.
- Pin converter source revision and enforce deterministic diffs in CI.

### Phase 3: Head/task coverage

- Add optional heads (MLM/classification) only after base-model load path is stable.

## Test Plan (for implementation PRs)

1. Config round-trip
- `DebertaRopeConfig.from_pretrained(...).to_dict()` preserves rope fields.

2. Auto-map resolution
- `AutoConfig.from_pretrained(..., trust_remote_code=True)` resolves `deberta-rope`.
- `AutoModel.from_pretrained(..., trust_remote_code=True)` instantiates `DebertaRopeModel`.

3. Weight parity
- Load exported discriminator weights into custom HF model and compare forward outputs against in-repo rope backbone on identical inputs.

4. Offline local load
- Loading works from a local export directory with no network calls.

5. Export contract safety
- Existing discriminator-only export path remains unchanged when code-emission is disabled.

## Open Decisions

- Whether code-emission should be opt-in (`--include-custom-code`) or a dedicated export mode.
- Whether generated HF runtime files should live in `src/deberta/hf_rope/` or `tools/codegen/outputs/`.
- Minimum supported `transformers` version for `auto_map` + custom classes.
