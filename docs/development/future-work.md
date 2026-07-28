# Future work

Parked improvements that are out of scope for the current branch but worth revisiting. Each entry states why it matters and what picking it up requires.

## Decide whether flagship configs should enable cross-document attention blocking

Every shipped config sets `data.packing.block_cross_document_attention: false` (matching the reference default), so the doc-block machinery - pairwise doc masks, per-document CLS conditioning via `doc_context_index`, document-local `position_ids`, and the flash `docblock`/`docblock_bias` kernels - is dormant in every shipped run, including the validated 50k FlashDeBERTa run. The two objective-side consequences of running unblocked (first-CLS RTD conditioning and row-continuous EMD positions) are described in the [data pipeline](../guides/data-pipeline.md#cross-document-attention-blocking); both are fixed by the `true` branch.

Naive packing is standard practice and may well be the right call (cross-document attention acts as benign noise at scale; blocking costs mask/metadata overhead), but today the setting reads as a default rather than a decision. Picking this up means: A/B a short run with blocking on (the flash `docblock` route keeps the cost story reasonable), compare discriminator metrics per document position within packed rows, and either flip the flagship configs or record here why `false` wins.

## Restrict EMD query axis to masked positions (exact, ~25% generator compute savings)

`EnhancedMaskDecoder` (`src/deberta/modeling/rtd.py`) reruns the last generator layer twice over the full sequence, then gathers only the masked positions (~15% of tokens) from the final pass. For a 6-layer generator, those two extra full-width passes add roughly 1/3 extra generator compute per step.

Restricting the query axis to masked positions across both EMD passes is mathematically exact, not an approximation:

- KV stays pinned to the penultimate hidden states in both passes (matching the original DeBERTa convention already implemented here).
- Pass-2 queries are pass-1 outputs row-wise: a masked position's pass-2 output depends only on its own pass-1 query row plus the full, unchanged KV.
- The residual/FFN in `DebertaV2Layer` operate per-position.

Only the masked query rows ever contribute to the MLM loss, so the remaining query-side attention and FFN work in the two extra passes is dead compute. Expected effect: EMD overhead drops from ~33% of generator cost to ~5-8% at `mlm_probability=0.15` with a 6-layer generator.

What it takes:

- Gather `z_states + kv_states` (and pass-1 outputs) on the query axis at `masked_idx` before each pass.
- Slice `relative_pos` on the query axis (rows of the `(Q,K)` matrix at masked positions); the KV axis stays full.
- Slice pairwise/broadcast attention masks on the query axis, including doc-block masks.
- Flash routes do not support ragged query subsets, so route EMD through the eager `DisentangledSelfAttention` path initially (it already accepts `query_states` with `query_len != key_len`), or add a varlen-query kernel later.
- Add a dedicated parity test: full-sequence EMD vs masked-query EMD must match to tight tolerance for eager, padded, and doc-packed batches. This is an improvement over original-DeBERTa behavior (the original has the same inefficiency), so parity coverage is what makes it safe.

Non-goals: no change to EMD semantics, pass count, or which positions are supervised.

## Keep the vendored positional-gradient addend in fp32 (upstream kernel change)

The repo already accumulates positional gradients into fp32 destinations (see
[FlashDeBERTa attention](../advanced/flash-attention.md#positional-gradient-accumulation-in-fp32)),
which removed the dominant order-sensitive accumulation loss. What remains: the pinned
`flashDeBERTa==0.0.7` backward kernels round `ds * sm_scale` to the model dtype *before* the
positional-gradient `tl.atomic_add` (`flashdeberta/ops/flash_attention.py:781`;
`flash_attention_varlen.py:915`), one rounding per contribution. Fixing that needs the kernels to
skip the pre-atomic downcast - an upstream PR or vendoring the two backward kernels. Low value on
its own (~2^-9 relative, unbiased); pick up only if the kernels get vendored for another reason.

Verified non-issue while auditing this: the atomics' `scope="cta"` in the fixed-route
kernels is correct - `DKPOS`/`DQPOS` writes are always CTA-local (grid axis 0 partitions the
written rows; batch/head base offsets are disjoint), and the varlen kernel, whose combined
kv-side atomics do collide across CTAs, uses the default `"gpu"` scope.

## Evaluate Adam-atan2 to drop eps tuning in dual-model RTD training

Dual-model RTD training (generator + discriminator, GDES sync, decoupled optimizers) is finicky: the two backbones have different gradient scales, and the backbone profiles already disagree on Adam epsilon (`hf_deberta_v2`: 1e-6, `rope`: 1e-8) - evidence that eps is a live tuning knob here rather than a solved constant.

Adam-atan2 ("Scaling Exponents Across Parameterizations and Optimizers", Everett et al., arXiv:2407.05872) replaces Adam's `m / (sqrt(v) + eps)` update with `atan2(m, sqrt(v))`, which:

- removes the eps hyperparameter entirely, along with its scale-sensitivity across model widths and backbones;
- is bounded, so tiny-`v` states cannot produce update blow-ups - relevant to the non-finite-window skip machinery the training loop maintains;
- would let both backbone profiles share one optimizer config instead of per-profile eps values.

Scope when picked up:

- Add as an opt-in `optim` choice next to the current AdamW path; keep AdamW the default.
- Verify interaction with FSDP2 sharded state and the GDES delta-embedding parameter group.
- A/B on a short pretraining run: loss curves, discriminator accuracy, and non-finite skip counts vs the current profiles.
