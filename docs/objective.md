# Objective: RTD (Replaced Token Detection)

This code implements the DeBERTaV3 / ELECTRA-style pretraining objective.

## Steps

1. **Mask tokens** in the input sequence (MLM-style).
2. Run the **generator** (masked LM) to predict masked tokens.
3. **Sample** replacements from the generator distribution.
4. Create a corrupted sequence by inserting sampled tokens at masked positions.
5. Run the **discriminator** on the corrupted sequence.
6. Train discriminator to predict whether each token is original (0) or replaced (1).

## Loss

Total loss:

- Generator loss: cross entropy over masked positions.
- Discriminator loss: BCEWithLogits over all non-padding tokens.

The trainer exposes knobs:

- `--disc_loss_weight`
- `--gen_loss_weight`
- `--sampling_temperature`
- `--decoupled_loss_scaling` (scales generator loss to be comparable to discriminator loss)

## Why masked-only logits matters

DeBERTa-v2/v3 uses a **large vocab** (often 128k). Computing `(batch, seq, vocab)` logits for the generator is extremely memory-heavy.

We therefore compute generator logits **only at masked token positions**, which keeps compute/memory practical.
