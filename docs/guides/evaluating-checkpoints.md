# Evaluating checkpoints

`tools/evaluate_rtd_checkpoint.py` scores committed training checkpoints on a deterministic
alternate shuffle of their own training source. Use it to check whether a run is learning, compare
checkpoints along one trajectory, or confirm a resumed run still tracks its history - no separate
evaluation dataset required.

```bash
python tools/evaluate_rtd_checkpoint.py \
    runs/<project>/<run>/checkpoint-5000 \
    runs/<project>/<run>/checkpoint-50000 \
    --output local-scratch/<run>/eval.json
```

Flags: `--output` (required JSON report path), `--batches` (default `4`), `--seed` (default
`20260720`), `--precision` (`bf16` default, or `fp32`).

## How it evaluates

- All listed checkpoints must belong to the same run. Model, data, and training settings come from
  the run-owned config snapshots and materialized tokenizer, so evaluation does not depend on the
  original config file, local paths, or Hub assets remaining unchanged.
- Attention is always eager with FlashDeBERTa disabled, so flash-trained and eager-trained
  checkpoints are scored by the same attention function. CUDA is required.
- The evaluation batch is drawn once from an alternate shuffle of the training source using
  `--seed` and `--batches`, then reused for every checkpoint. With a fixed seed, numbers are
  directly comparable across checkpoints of the same run.
- The evaluator masks its batch with the current collator. If the collator's masking behavior
  changes, rescore every side of a comparison; old and new reports are not comparable.
- Pick `--precision` to match the run's `train.mixed_precision`; a mismatch prints a warning to
  stderr.

## Reading the report

One JSON line per checkpoint is printed as it completes; the `--output` file adds full detail:
evaluation provenance and data description at the top level, then a per-step entry with `load`
(state-dict load stats), `state` (tensor count and non-finite element count), `generator`, and
`discriminator` blocks.

The useful signals:

- `generator.cross_entropy` - masked-token MLM loss; should fall as training progresses.
- `generator.replacement_rate` - fraction of masked positions where the generator sampled a
  replacement. This is the discriminator's positive-class prior; expect roughly 7-10% once the
  generator is competent.
- `discriminator.loss_gain` - BCE improvement over the constant class-prior predictor
  (`prior_bce - bce`). Positive gain means the discriminator carries real signal; a collapsed
  discriminator sits at or below zero.
- `discriminator.roc_auc` and `discriminator.average_precision` - ranking quality. Compare
  `average_precision` against `positive_rate`, which is the random-ranking baseline.
- `discriminator.token_centered_rms` - dispersion of active-token representations. Values near
  zero indicate representation collapse rather than a learned encoder.
- `state.nonfinite_elements` - must be zero; anything else means a corrupted checkpoint.

Replaced tokens are a minority class, so `accuracy_at_zero` is dominated by the majority class -
an all-original predictor already scores roughly 90%. Treat it (and the other `*_at_zero` fields)
as calibration diagnostics, not quality measures.

The thresholds this repo used to accept its production runs, and worked examples of these metrics
over a full 50,000-step trajectory, are in the
[RTD validation ledger](../development/rtd-validation-ledger.md#acceptance-criteria).
