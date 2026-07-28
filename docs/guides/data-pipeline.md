# Data pipeline

## Dataset source selection

`DataConfig` checks sources in this order:

1. `load_from_disk` (requires `streaming=false`)
2. `dataset_name` (with optional `dataset_config_name` and `data_files`)
3. `data_files` (HF `text` builder)

If none are provided, config validation fails.

Map-style sources (`data.source.streaming=false`, including `load_from_disk`) are explicitly partitioned across the combined distributed-rank and DataLoader-worker space. HF iterable sources (`data.source.streaming=true`) receive distributed-rank sharding from the dataset wrapper and retain their native DataLoader-worker partitioning.

Dataset loading and streaming make at most three attempts for transient I/O failures, with exponential backoff between the two retries. Retried streams replay to the last yielded example with the same epoch seed and shard; schema, authentication, and other non-transient errors fail immediately.

## Packed streaming path

`PackedStreamingDataset` (default when `data.packing.enabled=true`) tokenizes documents without
automatic special tokens. Its row structure depends on document blocking:

- With `block_cross_document_attention=false`, it inserts `[SEP]` between source documents, fills
  `max_seq_length - 2` content slots, and wraps the whole row as `[CLS] ... [SEP]`.
- With `block_cross_document_attention=true`, it first chunks each source document independently,
  wraps every chunk as `[CLS] document [SEP]`, then greedily packs whole wrapped segments. Segment
  chunking never depends on the remaining space in another row.

Outputs:

- `input_ids`
- `special_tokens_mask`
- optional `attention_mask` (only when padding exists)
- `doc_ids (B,S)` for document-blocked packing

Both dataset wrappers emit exactly `data.packing.max_seq_length` positions. Incomplete rows append
the tokenizer's `pad_token_id` on the right and emit a prefix-shaped attention mask. This setting is
the sole training row-length and padding target; choose a hardware- and kernel-friendly length
directly rather than combining it with a second training `pad_to_multiple_of` control.
For unblocked packing, a structural separator is inserted only when two documents share a row; the
row's outer SEP already supplies the boundary when the next document starts in a new row.

## Cross-document attention blocking

`data.packing.block_cross_document_attention` controls attention masking across packed document
boundaries. It is only valid with `data.packing.enabled=true`.

- `false`: packed samples attend across document boundaries; no document mask is built. Two
  objective-side consequences follow: the RTD head conditions every token on the row's *first*
  CLS, and EMD absolute positions continue across document boundaries instead of restarting.
  Both are the standard naive-packing tradeoff; the shipped configs currently accept it
  (see `docs/development/future-work.md` for the open decision).
- `true`: packed samples attend only within document segments. Every segment begins with its own CLS token.

The collator also emits two fixed-shape objective tensors for blocked rows:

- `position_ids (B,S)` restarts at zero for each document segment, so EMD sees the same learned
  absolute positions whether a document is standalone or packed at a later row offset.
- `doc_context_index (B,S)` maps each token to its document's CLS position. The RTD classifier
  gathers that context before LayerNorm, preserving the standalone `LayerNorm(token + CLS)`
  architecture for every packed document.

The collator derives contiguous document starts and ends once in row-major order. The objective tensors and FlashDeBERTa segment descriptors consume those same boundaries.
Within each row, every nonzero `doc_id` must occupy exactly one contiguous segment. The collator rejects reused IDs because dense equality masks would otherwise disagree with ragged segment metadata.

When FlashDeBERTa is enabled, the collator additionally precomputes fixed-capacity segment descriptors and scalar route inputs before device transfer. Eager training does not emit the Flash-only metadata.

What consumes `doc_ids` depends on the attention path:

- eager attention (both backbones) and the dense flash `docblock_bias` route materialize a
  boolean `(B,S,S)` pairwise keep-mask
- the flash ragged `docblock` route keeps the 2D `(B,S)` keep-mask and expands `doc_ids` into
  fixed-shape segment descriptors instead of a dense mask

Batch preparation is a second stage. `prepare_flash_attention_batch_metadata` consumes `doc_ids`,
chooses the attention route, and either materializes the pairwise mask or packages the collator's
`FlashBatchMeta`. External consumers must preserve `_flash_meta`, `position_ids`, and
`doc_context_index` through device transfer and call this function before the forward pass.
Preparation consumes `_flash_meta` while leaving the two objective tensors in the batch.

`attention_mask` is the only token-liveness authority. A token whose numeric ID equals
`pad_token_id` remains active when its mask value is true, and an arbitrary non-pad filler remains
inactive when its mask value is false. Token IDs identify document separators only at active
positions. Omitting `attention_mask` means every position is active.

Model forwards reject a raw `doc_ids` argument. Manually dropping `doc_ids` before batch preparation
loses the cross-document attention graph. Conversely, a pairwise/doc-block attention graph without
`doc_context_index` fails in the RTD head rather than silently dropping CLS conditioning.

Flash route selection for packed doc-block batches is described in
[Advanced / FlashDeBERTa attention](../advanced/flash-attention.md).

## Collator masking behavior

`DebertaV3ElectraCollator` applies dynamic MLM masking:

- `train.objective.mlm_max_ngram=1`: windowed unigram masking (DeBERTa's windowed selection, the
  parity path - not BERT-style iid masking)
- `train.objective.mlm_max_ngram>1`: whole-word n-gram masking

Mask replacement controls:

- `train.objective.mask_token_prob`
- `train.objective.random_token_prob`

Replacement probabilities are conditional on token selection and must sum to at most one; the remainder keeps the original token.

Masking uses `attention_mask` liveness captured before padding metadata is simplified. Inactive and special positions are never corruption targets and keep `labels=-100`. Each row's target budget is computed from its eligible lexical tokens, so padding and the number of packed CLS/SEP boundaries cannot change the lexical corruption rate. Whole-word masking applies one complete selected n-gram before checking the budget, so it may overshoot the nominal target by that n-gram's remaining tokens.

Windowed unigram selection sizes its windows from that budget: the candidate positions are split into exactly `num_to_predict` contiguous windows of near-equal size and one position is drawn uniformly from each. This keeps the selection count exact and every candidate position equally likely. Sizing windows from `int(1 / mlm_probability)` instead over-produces whenever the reciprocal is not an integer, and trimming the sorted excess back to the budget silently leaves the tail of every sequence unmasked.

Whole-word n-gram selection walks context windows of `n * int(1 / mlm_probability)` word groups. Special tokens split those windows and sampled spans, so an n-gram never crosses a packed-document boundary. Every partial context window is taken with probability proportional to its width; if all are rejected, one word group is selected uniformly across the row. Independently sampled n-grams remain separate budget units and are consumed in random order when stopping at the budget. These rules keep the mask rate positionally uniform without privileging the first document in a packed row.

Unmasked labels are `-100`; masked labels keep original token ids.

## Performance path for unpadded batches

For fixed-length packed batches without padding, the pipeline avoids creating all-ones `attention_mask` tensors and allows `attention_mask=None` fast paths downstream.
