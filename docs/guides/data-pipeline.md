# Data pipeline

## Dataset source selection

`DataConfig` checks sources in this order:

1. `load_from_disk` (requires `streaming=false`)
2. `dataset_name` (with optional `dataset_config_name` and `data_files`)
3. `data_files` (HF `text` builder)

If none are provided, config validation fails.

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
- `doc_ids` for document-blocked packing

## Cross-document attention blocking

`data.packing.block_cross_document_attention` controls attention masking across packed document
boundaries. It is only valid with `data.packing.enabled=true`.

- `false`: packed samples attend across document boundaries; no document mask is built
- `true`: the collator emits compact `doc_ids (B,S)` and precomputes fixed-capacity segment
  descriptors plus host statistics before device transfer. It validates exact, non-overlapping
  segment coverage against `attention_mask` before attesting the metadata. Every segment begins
  with its own CLS token.

The collator also emits two fixed-shape objective tensors for blocked rows:

- `position_ids (B,S)` restarts at zero for each document segment, so EMD sees the same learned
  absolute positions whether a document is standalone or packed at a later row offset.
- `doc_context_index (B,S)` maps each token to its document's CLS position. The RTD classifier
  gathers that context before LayerNorm, preserving the standalone `LayerNorm(token + CLS)`
  architecture for every packed document.

What consumes `doc_ids` depends on the attention path:

- eager attention (both backbones) and the dense flash `docblock_bias` route materialize a
  boolean `(B,S,S)` pairwise keep-mask
- the flash ragged `docblock` route keeps the 2D `(B,S)` keep-mask and expands `doc_ids` into
  fixed-shape segment descriptors instead of a dense mask

Batch preparation is a second stage. `prepare_flash_attention_batch_metadata` consumes `doc_ids`,
chooses the attention route, and either materializes the pairwise mask or packages the collator's
segment metadata into `FlashBatchMeta`. External consumers must preserve the collator's `flash_*`
fields, `position_ids`, and `doc_context_index` through device transfer and call this function
before the forward pass. A device batch that has lost required segment metadata fails rather than
rebuilding it on the GPU. Preparation consumes the raw Flash fields while leaving the two objective
tensors in the batch.

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

Unmasked labels are `-100`; masked labels keep original token ids.

## Performance path for unpadded batches

For fixed-length packed batches without padding, the pipeline avoids creating all-ones `attention_mask` tensors and allows `attention_mask=None` fast paths downstream.
