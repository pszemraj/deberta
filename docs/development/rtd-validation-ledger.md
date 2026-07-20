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

### Streaming preflight shutdown

Status: fixed and validated.

The first staged dry-run passed every reported preflight check but aborted during interpreter shutdown with `PyGILState_Release` from PyArrow. The failure reproduced outside the sandbox with a minimal `datasets.load_dataset(..., streaming=True)` iterator, so it was not a CUDA or sandbox restriction. The environment was Python 3.11.15, Datasets 3.6.0, and PyArrow 21.0.0.

Explicitly releasing and collecting an unshuffled iterator exited cleanly. Reservoir shuffle with `buffer_size=10000` still reproduced the abort during immediate process exit. Dry-run now samples one unshuffled source row because order is irrelevant to configuration/tokenization preflight, then explicitly closes and collects the iterator. Actual training still uses the configured shuffle unchanged. The checkpoint evaluator also releases its main-process data sampler before model evaluation.

The targeted CLI and evaluator suites pass (`94 passed`). The exact FlashDeBERTa dry-run and a shuffled one-batch checkpoint evaluation both exit zero outside the sandbox.

The CLI also intentionally rejected the attempted dotted command-line overrides, so staged runs use explicit YAML files rather than hidden runtime mutations.

Decision: the data-source preflight and evaluator lifecycle are no longer blockers. Proceed to the 100-step linear-decay smoke.

### Linear-decay 100-step smoke

Status: runtime passed; convergence gate not expected and not met.

This run compressed the production template to 100 steps with two warmup steps, preserving its 2% warmup ratio. All other model, data, batch, precision, and `1e-4` peak-LR settings matched `configs/flashdeberta/pretrain_flashdeberta_1024.yaml`. Configuration and outputs are under `local-scratch/flashdeberta-linear-smoke-100/`.

Training completed in 89 seconds at approximately 45k tok/s after startup. Checkpoints 50 and 100 saved cleanly, no non-finite events occurred, and strict discriminator export of checkpoint 100 succeeded.

| Step | Generator CE | Replacement rate | Discriminator BCE | Gain over prior | ROC-AUC | AP | Token RMS |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 50 | 7.671 | 14.90% | 0.3937 | 0.0272 | 0.682 | 0.280 | 0.538 |
| 100 | 7.458 | 14.89% | 0.4108 | 0.0099 | 0.611 | 0.221 | 0.521 |

All 315 state tensors were finite. The step-50 ranking signal and large representation dispersion rule out collapse, but the run is too short to meet the final acceptance criteria. Regression from step 50 to 100 coincided with the deliberately compressed scheduler approaching zero LR; it is not the prior numerical-collapse signature.

Decision: proceed to a 500-step linear stage before committing GPU time to 3,000 steps. Evaluate both the midpoint and endpoint because a compressed linear schedule is not an exact prefix of the 50k production schedule.

### Linear-decay 500-step stage

Status: passed as an intermediate stage; final convergence gate not yet met.

This run used 10 warmup steps and linear decay over 500 total steps, again preserving the production template's 2% warmup ratio. It completed 16.38M input tokens in 6m21s, with steady-state throughput near 45k tok/s. Both checkpoints saved, all state tensors were finite, and strict endpoint export succeeded.

| Step | Generator CE | Replacement rate | Discriminator BCE | Gain over prior | ROC-AUC | AP | Token RMS |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 250 | 6.390 | 13.83% | 0.3817 | 0.0201 | 0.612 | 0.197 | 0.573 |
| 500 | 5.973 | 13.37% | 0.3673 | 0.0261 | 0.645 | 0.232 | 0.628 |

Unlike the 100-step smoke, the endpoint improved over the midpoint despite LR reaching zero. AP at step 500 is 1.74 times its realized prior, and representation dispersion continues to grow. The run is non-degenerate and learning coherently but remains below the final `0.70` AUC, `2x`-prior AP, and `0.03` gain thresholds.

Decision: proceed to the committed 3,000-step recipe at `configs/flashdeberta/validate_flashdeberta_1024_3k.yaml`. It preserves the 2% warmup ratio with 60 warmup steps and saves checkpoints at 1k, 2k, and 3k.

### Linear-decay 3,000-step validation

Status: passed all acceptance criteria.

The committed validation recipe preserves the production model, data, effective batch size 32, BF16, GDES, peak learning rate `1e-4`, and linear scheduler. It compresses the production schedule from 50,000 steps with 1,000 warmup steps to 3,000 steps with 60 warmup steps, preserving the 2% warmup ratio. The run completed 98.30M input tokens in 36m52s at steady-state throughput near 45k tok/s. Checkpoints 1k, 2k, and 3k saved cleanly with no non-finite events.

The deterministic 16-sequence alternate-shuffle evaluation improved at every checkpoint:

| Step | Generator CE | Replacement rate | Discriminator BCE | Gain over prior | ROC-AUC | AP | Token RMS |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1,000 | 5.072 | 12.52% | 0.3286 | 0.0485 | 0.731 | 0.325 | 0.683 |
| 2,000 | 4.613 | 12.08% | 0.2842 | 0.0844 | 0.814 | 0.454 | 0.760 |
| 3,000 | 4.462 | 11.97% | 0.2775 | 0.0888 | 0.820 | 0.469 | 0.794 |

All 315 state tensors, containing 205,678,593 elements, were finite at every checkpoint. At 3k, AP was 3.92 times the realized positive prior. Accuracy at a zero logit threshold was 89.7%, but recall was only 19.0%; this again confirms that ranking and prior-relative loss, rather than raw zero-threshold accuracy, are the useful measurements.

A larger confirmation over 64 sequences and 65,536 active tokens measured generator CE `4.362`, replacement rate `0.1187`, discriminator BCE `0.2787`, prior gain `0.0857`, ROC-AUC `0.8182`, AP `0.4579`, logit standard deviation `3.30`, and token-centered RMS `0.788`. The result therefore does not depend on the smaller 16-sequence sample.

Strict discriminator export of checkpoint 3k completed and loaded through `AutoModel`. Native-checkpoint versus standalone-export comparison over 26,880 active hidden-state elements measured maximum absolute error `3.78e-6`, mean absolute error `4.89e-7`, and cosine similarity `0.99999994`. The first two disposable parity-script attempts failed before model execution because the repository root was absent from the script import path and then because the tokenizer field was read from `data` instead of `model`; removing the unnecessary tool import and correcting the field resolved both. No tracked implementation change was needed.

Decision: accept checkpoint 3k as the first end-to-end non-degenerate model produced by this codebase. The linear `1e-4` recipe clears the convergence, representation, finiteness, export, and parity gates. Proceed with `configs/flashdeberta/pretrain_flashdeberta_1024.yaml` for a longer production-scale run while continuing to monitor held-out prior gain, ROC-AUC, and AP at saved checkpoints.

## Next iteration

1. Start the 50,000-step production recipe only when that longer GPU allocation is desired; the 3,000-step convergence gate no longer blocks it.
2. Evaluate each 5,000-step production checkpoint with the tracked alternate-shuffle evaluator and retain downstream mean-pooling probes as a representation-quality cross-check.
3. Treat a material fall in held-out ranking or representation dispersion, rather than zero-threshold accuracy, as the signal to pause and investigate.
