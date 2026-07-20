# RTD validation ledger

This is the chronological engineering record for bringing native DeBERTa-v3 RTD pretraining to a demonstrably non-degenerate checkpoint. It records the experiment intent, exact configuration differences, decisive measurements, failures, and resulting decisions. Generated logs and evaluation JSON stay under `local-scratch/` and are not committed.

## Acceptance criteria

A validation checkpoint is considered non-degenerate when all of the following hold on an alternate shuffle of the training source:

- every checkpoint tensor is finite;
- generator masked-token cross-entropy improves from early training;
- discriminator loss beats the positive-prior entropy by at least `0.03` BCE;
- discriminator ROC-AUC is at least `0.70` and average precision is at least twice the realized replacement prior;
- active-token representations have token-centered RMS above `1e-3`, which cleanly separates a learned encoder from the previously observed numerical collapse;
- strict Hugging Face discriminator export succeeds and active-token outputs match the native checkpoint within `1e-4` maximum absolute error.

Raw discriminator accuracy is not an acceptance metric. Replacements are a minority class, so an all-original prediction already produces roughly 90% accuracy. Loss gain over the realized class prior, ROC-AUC, and average precision are the primary RTD measurements.

## Experiment ledger

### Previous 50k run: discriminator collapse

Status: failed.

The earlier native DeBERTa run used a `5e-4` learning rate. Its discriminator converged to the replacement-class prior rather than learning token distinctions: held-out ROC-AUC was approximately `0.50`, while representation and logit variation were approximately `1e-7`. This established a concrete collapse signature and motivated comparison with the original Microsoft recipe.

Decision: use the original base-model learning rate of `1e-4`. Commit `451dcc6` made that the native DeBERTa default and documented the failure mode.

### Constant-LR 12k stability run

Status: passed as a stability test.

Configuration and outputs are under `local-scratch/flashdeberta-validation-12k/`. The run used FlashDeBERTa at sequence length 1024, effective batch size 32, BF16, GDES, 1,000 warmup steps, and then held `1e-4` constant through 12,000 steps. This was intentionally harsher than the intended linear-decay production schedule.

Evaluation used 16 alternate-shuffle sequences from `EleutherAI/SmolLM2-1.7B-stage-4-100B`, totaling 16,384 tokens:

| Step | Generator CE | Replacement rate | Discriminator BCE | Gain over prior | ROC-AUC | AP |
|---:|---:|---:|---:|---:|---:|---:|
| 3,000 | 3.98 | 11.5% | 0.279 | 0.078 | 0.808 | 0.422 |
| 6,000 | 3.19 | 10.1% | 0.267 | 0.061 | 0.793 | 0.337 |
| 9,000 | 2.86 | 9.4% | 0.256 | 0.056 | 0.793 | 0.299 |
| 12,000 | 2.71 | 9.1% | 0.250 | 0.054 | 0.793 | 0.296 |

All 315 checkpoint tensors were finite. At 12k, discriminator logit standard deviation was `3.46` and token-centered representation RMS was `0.89`. The F1-optimal discriminator logit threshold was approximately `-1.63`; the default zero threshold had only 5.2% recall and made raw accuracy look uninformative.

A frozen linear SST-2 probe over mean-pooled representations improved from 63.4% at initialization to 69.3% at 12k. CLS-only performance remained mostly flat. Wikitext transfer retained ranking signal (`0.620` ROC-AUC and `0.243` AP against a `0.093` prior) but was poorly calibrated. Random-substitution detection peaked around 6k while SST-2 continued improving, consistent with late specialization under constant LR rather than collapse.

Decision: the `1e-4` correction is validated. Keep training beyond 6k and use linear decay for the intended production recipe.

### Original Microsoft DeBERTa comparison

Status: implementation contract corroborated.

The reference checkout is under `local-scratch/microsoft_DeBERTa/`. Its base recipe agrees with the current implementation on the 12-layer discriminator, 6-layer generator, GDES, enhanced mask decoder, RTD head ordering, categorical generator sampling, discriminator loss weight 10, `1e-4` learning rate, Adam epsilon `1e-6`, weight decay `0.01`, and gradient clipping at 1.

The major recipe differences are intentional: the reference uses sequence length 512, dropout 0.1, batch size 256, 10k warmup, and linear decay over one million steps. FlashDeBERTa requires zero dropout; the 12k run used length 1024, effective batch size 32, 1k warmup, and constant LR.

Decision: no core RTD-objective mismatch is currently indicated. Treat scheduling, scale, and held-out measurements as the next validation surface.

### Strict GDES export

Status: fixed and validated.

Strict discriminator export of checkpoint 12k retains `embeddings.position_embeddings.bias`, even though a standalone Hugging Face backbone configured with `position_biased_input=false` has no absolute-position embedding. A local runtime patch that drops GDES bias keys absent from the export model produced a 110M-parameter artifact that loaded through `AutoModel`. Active-token outputs matched with cosine approximately 1.0 and maximum absolute error `8.8e-6`.

The strict-load preparation now drops every GDES embedding bias key that is absent from the target export backbone, including the orphaned position bias. The focused export suites pass (`21 passed`). Strict export of the actual 12k checkpoint then completed without partial loading, and the resulting 110,026,752-parameter artifact loaded through `AutoModel`. The earlier runtime-patched parity check already measured active-token cosine approximately 1.0 and maximum absolute error `8.8e-6`, within the acceptance threshold.

Decision: strict GDES export is no longer a blocker. Keep strict loading as the artifact acceptance path.

### Tracked alternate-shuffle evaluator

Status: passed.

`tools/evaluate_rtd_checkpoint.py` now evaluates one or more checkpoints on a deterministic alternate shuffle of the configured training source. It reports generator CE and realized replacement rate; discriminator BCE, prior gain, ROC-AUC, AP, zero-threshold behavior, and logit dispersion; representation token-centered RMS; and checkpoint finiteness. The ROC-AUC and AP implementations are dependency-free and covered by hand-calculated and tied-score tests (`4 passed`).

The tracked evaluator reproduced the established 12k result over 16,384 tokens: generator CE `2.709`, replacement rate `0.0909`, discriminator BCE `0.2481`, loss gain `0.0565`, ROC-AUC `0.7991`, AP `0.3080`, token-centered RMS `0.8880`, and zero non-finite elements across all 315 state tensors. The generated JSON remains under `local-scratch/flashdeberta-validation-12k/`.

Decision: use this command after staged checkpoints so run decisions use the same deterministic measurement path.

## Next iteration

1. Validate the existing FlashDeBERTa production template (`1e-4`, linear decay) with short smoke runs.
2. Run a 1,000–3,000-step staged validation, evaluate against the criteria above, and verify strict export.
