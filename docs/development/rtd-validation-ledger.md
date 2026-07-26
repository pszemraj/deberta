# RTD validation ledger

Native DeBERTa-v3 RTD validation progressed from a collapsed discriminator to a non-degenerate 50,000-step checkpoint. Generated logs and evaluation JSON stay under `local-scratch/` and are not committed.

## Acceptance criteria

A validation checkpoint is considered non-degenerate when all of the following hold on an alternate shuffle of the training source:

- every checkpoint tensor is finite;
- generator masked-token cross-entropy improves from early training;
- discriminator loss beats the positive-prior entropy by at least `0.03` BCE;
- discriminator ROC-AUC is at least `0.70` and average precision is at least twice the realized replacement prior;
- active-token representations have token-centered RMS above `1e-3`, which separates a learned encoder from the observed numerical collapse;
- strict Hugging Face discriminator export succeeds and active-token outputs match the native checkpoint within `1e-4` maximum absolute error.

Raw discriminator accuracy is not an acceptance metric. Replacements are a minority class, so an all-original prediction already produces roughly 90% accuracy. Loss gain over the realized class prior, ROC-AUC, and average precision are the primary RTD measurements.

## Validation progression

| Stage | Purpose and result | Decision |
|---|---|---|
| Earlier 50k run | A `5e-4` learning rate collapsed the discriminator to roughly `0.50` ROC-AUC with representation and logit variation near `1e-7`. | Use the Microsoft base-model learning rate of `1e-4`. |
| Constant-LR 12k | A deliberately harsh constant `1e-4` schedule reached generator CE `2.71`, discriminator BCE `0.250`, prior gain `0.054`, ROC-AUC `0.793`, AP `0.296`, and token RMS `0.89`; all 315 state tensors were finite. | The corrected learning rate is stable; use linear decay for production. |
| Microsoft recipe comparison | The model dimensions, GDES, enhanced mask decoder, RTD head ordering, sampling, loss weighting, optimizer epsilon, weight decay, and clipping agree with the reference implementation. | No core RTD-objective mismatch was indicated. |
| Strict GDES export | Removing GDES bias keys absent from the standalone backbone allowed strict loading. Native and exported active-token outputs matched within `8.8e-6`. | Keep strict loading as the artifact acceptance path. |
| Evaluator and preflight | The tracked evaluator reproduced the 12k result. Explicit release of short-lived streaming iterators eliminated a PyArrow interpreter-shutdown abort in dry-run and evaluation. | Use the deterministic evaluator for staged checkpoints and retain explicit iterator cleanup. |
| Linear 100 and 500 | Both short stages were finite and non-degenerate, but their compressed schedules did not meet the convergence thresholds. | Continue to the committed 3,000-step validation recipe. |
| Linear 3k | Generator CE reached `4.462`, discriminator prior gain `0.0888`, ROC-AUC `0.820`, AP `0.469`, and token RMS `0.794`. Strict export matched within `3.78e-6`. | All acceptance criteria passed; proceed to 50,000 steps. |

## Linear-decay 50,000-step production run

Status: passed all acceptance criteria with useful downstream representations.

The tracked production recipe completed 1.638B input tokens in 10h14m17s at approximately 44.7k tok/s. All ten 5,000-step checkpoints committed successfully. The final training window reported generator CE `1.889`, discriminator BCE `0.189`, prior gain `0.0673`, and replacement rate `0.0710` as the linear scheduler reached zero.

The deterministic 16-sequence alternate-shuffle evaluation showed no late collapse. Generator CE, discriminator BCE, and ROC-AUC improve monotonically; AP and gain over prior stay flat, with AP ending at `0.323` against `0.328` at 5,000 steps:

| Step | Generator CE | Replacement rate | Discriminator BCE | Gain over prior | ROC-AUC | AP | Token RMS |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 5,000 | 3.351 | 10.36% | 0.2744 | 0.0586 | 0.787 | 0.328 | 0.883 |
| 10,000 | 2.819 | 9.39% | 0.2537 | 0.0579 | 0.800 | 0.323 | 0.906 |
| 15,000 | 2.603 | 9.01% | 0.2457 | 0.0572 | 0.804 | 0.307 | 0.890 |
| 20,000 | 2.475 | 8.71% | 0.2378 | 0.0580 | 0.810 | 0.312 | 0.883 |
| 25,000 | 2.387 | 8.45% | 0.2316 | 0.0579 | 0.816 | 0.305 | 0.886 |
| 30,000 | 2.315 | 8.39% | 0.2298 | 0.0583 | 0.816 | 0.311 | 0.884 |
| 35,000 | 2.239 | 8.21% | 0.2266 | 0.0573 | 0.816 | 0.309 | 0.890 |
| 40,000 | 2.201 | 8.09% | 0.2231 | 0.0580 | 0.822 | 0.305 | 0.886 |
| 45,000 | 2.170 | 8.07% | 0.2220 | 0.0586 | 0.823 | 0.312 | 0.888 |
| 50,000 | 2.147 | 8.09% | 0.2198 | 0.0611 | 0.828 | 0.323 | 0.887 |

Every checkpoint had zero non-finite elements across all 315 state tensors. A larger endpoint confirmation over 64 sequences and 65,536 active tokens measured generator CE `2.050`, replacement rate `0.0780`, discriminator BCE `0.2091`, prior gain `0.0649`, ROC-AUC `0.8424`, AP `0.3440`, logit standard deviation `4.12`, and token-centered RMS `0.890`. AP was 4.41 times the realized replacement prior. Zero-threshold recall was only 9.0%, so the 92.5% raw accuracy remains a calibration artifact rather than the useful quality measure.

The frozen 5,000-example SST-2 probe also improved beyond the 12k result:

| Checkpoint | CLS validation accuracy | Mean-pooled validation accuracy |
|---:|---:|---:|
| Fresh | 63.2% | 63.4% |
| 5,000 | 61.0% | 65.1% |
| 10,000 | 63.2% | 69.4% |
| 25,000 | 66.4% | 67.3% |
| 50,000 | 67.9% | 70.8% |

Mean pooling remains the preferred default for frozen features. The temporary mean-probe dip at 25k recovered to a new best at 50k, while token-level representation dispersion remained stable throughout training.

Wikitext transfer confirms a calibration limitation rather than collapse. Against replacements sampled by the 50k generator, the discriminator retained ROC-AUC `0.642` and AP `0.255` against a `0.085` prior, but BCE was `0.541` against prior entropy `0.291`. Fixed-random substitution similarly retained ranking signal (ROC-AUC `0.648`, AP `0.456` against a `0.148` prior) with poor raw calibration. Product use of RTD logits therefore needs a threshold or calibration fit on its target corruption and text distribution.

The automatic final Hugging Face artifact loaded through `AutoModel`. Comparison with checkpoint 50k over 26,880 active hidden-state elements measured maximum absolute error `8.52e-6`, mean absolute error `7.46e-7`, and cosine similarity indistinguishable from 1.0.

Decision: accept checkpoint 50k and its `final_hf` export as the successful production outcome. The generator improves, discriminator ROC-AUC strengthens through the end of training while AP holds flat, representations remain diverse, downstream mean-pooled quality improves, and export is faithful. Treat cross-domain RTD calibration as a downstream product concern, not a pretraining defect.

## Collator masking fix rerun

The run above trained on windowed-unigram masking that sized its windows from `int(1 / 0.15) = 6`. That over-produced candidates against the 15% budget and trimmed the sorted excess, so the last roughly 9.6% of every 1024-token row was never a masking target. The rerun repeats the same recipe on the fixed collator, differing only in logging, and is the reference production run from this point forward.

It completed 1.638B input tokens in 12h39m18s at approximately 36.0k tok/s. The throughput drop relative to the run above was a code regression, since isolated and fixed; see [Triton throughput regression](#triton-throughput-regression) below. It was not the collator change, which leaves the per-row masked-token count identical because the old path also trimmed its selection back to the same budget. All ten checkpoints and the `final_hf` export committed successfully, with zero non-finite elements. The final training window reported generator CE `1.791`, discriminator BCE `0.1945`, prior gain `0.0609`, and replacement rate `0.0707`.

Both runs' checkpoints were then scored under one identical evaluation: the same config, seed `20260720`, 16 sequences, 16,384 active tokens, and 2,464 masked tokens, with eager attention and flash disabled. Only the loaded weights differ.

| Step | Generator CE | Discriminator BCE | Gain over prior | ROC-AUC | AP |
|---:|---:|---:|---:|---:|---:|
| 5,000 | 3.382 -> 3.379 | 0.3817 -> 0.2816 | -0.0463 -> 0.0527 | 0.712 -> 0.773 | 0.298 -> 0.328 |
| 15,000 | 2.700 -> 2.697 | 0.3622 -> 0.2494 | -0.0551 -> 0.0522 | 0.726 -> 0.798 | 0.252 -> 0.296 |
| 25,000 | 2.459 -> 2.467 | 0.3416 -> 0.2402 | -0.0466 -> 0.0529 | 0.741 -> 0.803 | 0.259 -> 0.298 |
| 35,000 | 2.353 -> 2.336 | 0.3353 -> 0.2289 | -0.0472 -> 0.0587 | 0.745 -> 0.819 | 0.265 -> 0.331 |
| 50,000 | 2.259 -> 2.253 | 0.3317 -> 0.2220 | -0.0477 -> 0.0630 | 0.749 -> 0.834 | 0.280 -> 0.346 |

Generator cross-entropy is unchanged, so the fix neither eased nor hardened the MLM task. The discriminator carries the entire difference. Scored on positionally uniform masking, the old checkpoints land below the constant class-prior predictor at every stage; at 50,000 steps their logit standard deviation is `4.22` against the rerun's `2.71`, confidently wrong on the tail positions they never saw corrupted. The rerun also beats the old run's own numbers from its home distribution above (gain `0.0611`, ROC-AUC `0.828`, AP `0.323` at 50k), so the improvement is not an artifact of scoring the old weights off-distribution.

Two measurement notes for anyone rereading the older tables. The evaluator's `replacement_rate` now divides by masked-token count, whereas the 50k table above predates that change and reports the share of all tokens; the two differ by roughly the masking rate. And because the evaluator masks its own batches with the current collator, numbers produced before the fix are not comparable to numbers produced after it. Rescore both sides whenever the collator changes.

## Triton throughput regression

The 44.7k -> 36.0k tok/s drop between the two 50,000-step runs was a code regression, not the driver and kernel upgrade that happened to land between them. Both runs were rebenchmarked on one host and one driver with `configs/flashdeberta/bench_flashdeberta_1024.yaml`, which matches the production recipe in every field that affects per-step cost.

Measure steady state from `input_tokens_seen` deltas between steps 100 and 250. The trainer's own `tok/s` column is a cumulative average and is contaminated by startup and Triton autotune.

| Tree | tok/s |
|---|---:|
| `fc80aac`, before the regression | 43,497 |
| `ddabf47` through `ac4454a` | 35,877 |
| `81b46e9`, after the fix | 48,188 |

A binary search over the commits touching `src/` between those endpoints reached `ddabf47`, and reverting `flashdeberta_dense_bias_op.py` alone to its parent restored the throughput. Isolating the op showed the forward kernel itself at 1.235 ms against 0.209 ms for the same shapes and the same launch config, with byte-identical output; the compiled kernel had dropped from 32,768 bytes of shared memory to zero.

The cause is Triton argument specialization. Annotating a kernel parameter with a real type makes Triton canonicalize the annotation to `i32` or `fp32` and pin the argument type from it, instead of specializing on the value passed. That specialization is what tells Triton a stride is divisible by 16 or exactly 1, so losing it costs the compiler its contiguity proof and its vectorized loads. These annotations had been quoted, and under `from __future__ import annotations` a quoted `"int"` is stored as `"'int'"`, which never matches a canonical type name and so stayed inert. `ddabf47` rewrote them to real types and switched the mechanism on.

Six repo-defined kernels across five modules carried the same defect, 152 parameters in total. They are now annotated `None`, which Triton does not canonicalize; `test_triton_kernels_keep_runtime_value_specialization` fails if any of them regains a real type. The fixed tree beats the pre-regression baseline because `ddabf47` also added bucket bounds masks that removed 22 register spills, which is worth its 3% kernel cost.

## Sources

| Numbers | Artifact |
|---|---|
| 12k and 3k validation rows | `local-scratch/flashdeberta-validation-12k/`, `configs/flashdeberta/validate_flashdeberta_1024_3k.yaml` |
| 50,000-step production run | `runs/flashdeberta/20260720_205652_pretrain_flashdeberta_1024/`, `configs/flashdeberta/pretrain_flashdeberta_1024.yaml` |
| Per-checkpoint evaluation, SST-2 probe, Wikitext transfer, export parity | `local-scratch/flashdeberta-production-50k/` (`eval-history-4b.json`, `eval-final-16b.json`, `frozen-sst2.json`, `wikitext-transfer.json`, `export-parity.json`) |
| Collator-fix rerun and both-sides rescoring | `runs/flashdeberta/20260725_053746_flashdeberta-1024-50k-maskfix/`, `local-scratch/flashdeberta-1024-50k-maskfix/` (`train.log`, `eval-baseline-recheck.json`, `eval-maskfix.json`) |
| Throughput bisect | `local-scratch/throughput-bisect/` |

Checkpoint evaluations were produced with `tools/evaluate_rtd_checkpoint.py`.
