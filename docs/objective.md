# Objective: RTD (Replaced Token Detection)

See also: [data/collator](data.md), [runtime](fsdp2.md).

## RTD Flow

Each training step follows DeBERTaV3/ELECTRA-style RTD:

1. Apply dynamic MLM masking to input tokens.
2. Run generator on masked input.
3. Compute generator logits only at masked positions.
4. Sample replacements from generator distribution, excluding config special token ids (`pad/cls/sep/mask/bos/eos`) plus runtime-provided tokenizer special ids.
5. Create corrupted sequence by inserting sampled tokens at masked positions.
6. Run discriminator on corrupted sequence.
7. Train discriminator to predict original (`0`) vs replaced (`1`) per token.

When `train.decoupled_training=true`, this is executed as two optimizer phases per global
step (generator step, embedding sync for GDES, then discriminator step).

Special-token filtering in RTD is config-driven by default, with optional runtime additive ids from `tokenizer.all_special_ids` in training/export paths.

## Loss Terms

- Generator loss: cross entropy on masked positions only.
- Discriminator loss: BCE-with-logits on all active non-padding tokens.
- Total loss: `gen_loss_weight * gen_loss + disc_loss_weight * disc_loss`.

Exposed controls:

- `train.gen_loss_weight`
- `train.disc_loss_weight` (default `50.0`)
- `train.decoupled_loss_scaling`
- `train.decoupled_training` (`null` auto-resolves by backbone type)
- `train.sampling_temperature`
- `train.token_weighted_gradient_accumulation` (default `true`)

For parity-profile defaults and long-run parity configs, see [replication](replication.md).

Per-microbatch loss terms are token-level means. When gradient accumulation is enabled, token-weighted accumulation is used by default so each microbatch contributes proportionally to its active-token counts (instead of equal microbatch averaging).

For `model.embedding_sharing=gdes`, discriminator embedding base weights are synchronized from the
generator after each optimizer step (and checkpoint load). During a gradient-accumulation window,
the discriminator therefore sees the previous optimizer-step snapshot until the next sync.

## Numerical Stability

Both generator CE loss and discriminator BCE loss are computed from fp32 logits, even when training uses bf16 autocast.

## Why masked-only generator logits

Vocab logits for all tokens (`batch x seq x vocab`) are expensive with large DeBERTa vocabularies.

This implementation computes generator vocab projection only on masked positions to reduce memory/compute while preserving RTD behavior.
