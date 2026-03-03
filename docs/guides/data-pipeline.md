# Data pipeline

## Dataset source selection

`DataConfig` checks sources in this order:

1. `load_from_disk` (requires `streaming=false`)
2. `dataset_name` (with optional `dataset_config_name` and `data_files`)
3. `data_files` (HF `text` builder)

If none are provided, config validation fails.

## Packed streaming path

`PackedStreamingDataset` (default when `pack_sequences=true`) does:

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

`data.block_cross_document_attention` controls pairwise document masking for packed samples.

- `false`: no 3D pairwise mask path
- `true`: collator emits `doc_ids (B,S)` and training materializes a `(B,S,S)` keep-mask

This mode is only valid with `pack_sequences=true`.

## Collator masking behavior

`DebertaV3ElectraCollator` applies dynamic MLM masking:

- `train.mlm_max_ngram=1`: token-level masking (parity path)
- `train.mlm_max_ngram>1`: whole-word n-gram masking

Mask replacement controls:

- `train.mask_token_prob`
- `train.random_token_prob`

Unmasked labels are `-100`; masked labels keep original token ids.

## Performance path for unpadded batches

For fixed-length packed batches without padding, the pipeline avoids creating all-ones `attention_mask` tensors and allows `attention_mask=None` fast paths downstream.
