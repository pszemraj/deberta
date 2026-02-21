# Objective: RTD (Replaced Token Detection)

This document is the primary reference for the pretraining objective implemented in this repo.

For data/collator behavior, see [`docs/data.md`](data.md). For distributed/runtime configuration, see [`docs/fsdp2.md`](fsdp2.md).

## RTD Flow

Each training step follows DeBERTaV3/ELECTRA-style RTD:

1. Apply dynamic MLM masking to input tokens.
2. Run generator on masked input.
3. Compute generator logits only at masked positions.
4. Sample replacements from generator distribution, excluding configured special token ids when available (`pad/cls/sep/mask/bos/eos`).
5. Create corrupted sequence by inserting sampled tokens at masked positions.
6. Run discriminator on corrupted sequence.
7. Train discriminator to predict original (`0`) vs replaced (`1`) per token.

## Loss Terms

- Generator loss: cross entropy on masked positions only.
- Discriminator loss: BCE-with-logits on active non-special tokens.
- Total loss: `gen_loss_weight * gen_loss + disc_loss_weight * disc_loss`.

Exposed controls:

- `train.gen_loss_weight`
- `train.disc_loss_weight` (default `50.0`)
- `train.decoupled_loss_scaling`
- `train.sampling_temperature`

Per-microbatch loss terms are token-level means; gradient-accumulation windows currently weight microbatches equally.

## Numerical Stability

Both generator CE loss and discriminator BCE loss are computed from fp32 logits, even when training uses bf16 autocast.

## Why masked-only generator logits

Vocab logits for all tokens (`batch x seq x vocab`) are expensive with large DeBERTa vocabularies.

This implementation computes generator vocab projection only on masked positions to reduce memory/compute while preserving RTD behavior.

Deferred objective/loss follow-ups are tracked in [`docs/roadmap.md`](roadmap.md).
