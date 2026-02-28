# Data Pipeline

See also: [objective/loss](objective.md), [model/backbone](model.md), [runtime/FSDP2](fsdp2.md).

## Dataset Source Selection

`DataConfig` supports three input modes (checked in this order):

1. `load_from_disk` (non-streaming only)
2. `dataset_name` (+ optional `dataset_config_name` and `data_files`)
3. `data_files` (uses the `text` dataset builder)

`streaming=true` is the default and recommended for pretraining.

## Streaming Packing (Default)

`PackedStreamingDataset` consumes raw text and emits fixed-length packed examples:

- tokenizes documents with `add_special_tokens=False`
- appends an internal document separator (`[SEP]`) between documents
- packs into chunks of `max_seq_length - 2`
- wraps each chunk as `[CLS] + chunk + [SEP]` (+ pad when needed)
- flushes trailing remainder at iterator end (no silent tail-token drop)

Output fields:

- `input_ids` (always)
- `special_tokens_mask` (always)
- `attention_mask` (only when padding exists)

### Internal `[SEP]` handling

Packed chunks can contain inserted internal `[SEP]` separators. Those positions are marked as special in `special_tokens_mask`, so masking never corrupts those separators.

`data.block_cross_document_attention` controls whether packed batches use strict pairwise document-blocking:

- `false` (default): skip 3D doc-blocking masks (packed batches remain on 2D/no-mask attention paths)
- `true`: emit 3D doc-blocking masks for packed batches with internal separators

### Pairwise Mask Contract (`block_cross_document_attention=true`)

- mask type/shape: boolean keep-mask `(B, S, S)` where `True=attend`, `False=block`
- emission condition: only emitted when packed chunks contain internal separators; single-document packed chunks keep `attention_mask=None`
- query activity encoding: diagonal `True` marks active queries, diagonal `False` marks inactive/padded queries
- SDPA safety for inactive queries: inactive/padded query rows include a single keep edge to CLS key to avoid all-False rows in backend kernels; outputs remain zeroed via diagonal-based query activity
- document boundary behavior: CLS stays in document 1 (no global cross-document CLS channel)
- consumers: rope attention and RTD token-activity accounting use this contract; keep `data.block_cross_document_attention=false` when strict doc-blocking is unnecessary

## Sequential / No-Pack Mode

Set `data.pack_sequences=false` to switch to one-document chunking:

- no cross-document concatenation
- long documents are split into multiple one-document chunks
- useful as a reference mode for validating loss-signal behavior independent of packing
- `data.block_cross_document_attention=true` is invalid in this mode (validation error)

## Non-Streaming Packing

When `streaming=false`, training still routes through the same iterable packing wrappers (`PackedStreamingDataset` / `SequentialStreamingDataset`) used in streaming mode, but backed by a map-style HF dataset iterator.

`data.shuffle_buffer_size` keeps full buffer semantics only for streaming datasets. In non-streaming mode, shuffle is effectively an off/on toggle and validation requires `data.shuffle_buffer_size` to be `0` (off) or `1` (on).

## Collator Behavior (Dynamic MLM Masking)

`DebertaV3ElectraCollator` performs dynamic masking at batch time.

Masking modes:

- token-level masking: `train.mlm_max_ngram = 1` (default)
- whole-word n-gram masking: `train.mlm_max_ngram > 1`

Replacement probabilities are controlled by:

- `train.mask_token_prob`
- `train.random_token_prob`

Labels follow standard HF MLM semantics:

- masked positions: original token id
- unmasked positions: `-100`

## Attention Mask Optimization for SDPA/Flash

For packed, fixed-length, unpadded batches, explicit all-ones masks are redundant.

The pipeline keeps this path lean by:

1. Not emitting `attention_mask` in packed dataset outputs when no padding exists.
2. Asking tokenizer padding to skip mask materialization when possible.
3. Dropping all-ones `attention_mask` tensors in the collator.

Downstream model code treats missing masks as unpadded (`attention_mask=None`).

Deferred data/packing follow-ups are tracked in the [roadmap](roadmap.md).
