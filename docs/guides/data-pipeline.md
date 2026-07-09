# Data pipeline

## Dataset source selection

`DataConfig` checks sources in this order:

1. `load_from_disk` (requires `streaming=false`)
2. `dataset_name` (with optional `dataset_config_name` and `data_files`)
3. `data_files` (HF `text` builder)

If none are provided, config validation fails.

## Packed streaming path

`PackedStreamingDataset` (default when `data.packing.enabled=true`) does:

1. tokenize documents without auto special tokens
2. insert internal `[SEP]` between documents
3. pack tokens to `max_seq_length - 2`
4. wrap each sample as `[CLS] ... [SEP]`
5. pad only when needed

Outputs:

- `input_ids`
- `special_tokens_mask`
- optional `attention_mask` (only when padding exists)

## Cross-document attention blocking

`data.packing.block_cross_document_attention` controls attention masking across packed document
boundaries. It is only valid with `data.packing.enabled=true`.

- `false`: packed samples attend across document boundaries; no document mask is built
- `true`: the collator emits compact `doc_ids (B,S)` and precomputes fixed-capacity segment
  descriptors plus host statistics before device transfer

What consumes `doc_ids` depends on the attention path:

- eager attention (both backbones) and the dense flash `docblock_bias` route materialize a
  boolean `(B,S,S)` pairwise keep-mask
- the flash ragged `docblock` route keeps the 2D `(B,S)` keep-mask and expands `doc_ids` into
  fixed-shape segment descriptors instead of a dense mask

Batch preparation is a second stage. `prepare_flash_attention_batch_metadata` consumes `doc_ids`,
chooses the attention route, and either materializes the pairwise mask or packages the collator's
segment metadata into `FlashBatchMeta`. External consumers must preserve the collator's `flash_*`
fields through device transfer and call this function before the forward pass. A device batch that
has lost required segment metadata fails rather than rebuilding it on the GPU.

Model forwards reject a raw `doc_ids` argument. Manually dropping `doc_ids` and forwarding only
`input_ids` and `attention_mask` silently re-enables cross-document attention and global-CLS
conditioning in the RTD head; there is no in-model guard for that misuse.

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
