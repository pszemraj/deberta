# Data API

## Module `deberta.data`

Data utilities for DeBERTaV3 pretraining.

## `DebertaV3ElectraCollator`

```python
class DebertaV3ElectraCollator(
    *,
    tokenizer: 'Any',
    cfg: 'MLMConfig',
    packed_sequences: 'bool' = False,
    block_cross_document_attention: 'bool' = True,
    pad_to_multiple_of: 'int | None' = None,
) -> 'None'
```

Dynamic MLM masking collator suitable for RTD/ELECTRA-style pretraining.

Produces masked ``input_ids``, MLM ``labels``, and optional attention/token-type tensors.
Packed doc-block batches also carry ``doc_ids`` and ``flash_*`` metadata. See
[Data pipeline](../guides/data-pipeline.md#cross-document-attention-blocking) for the complete
batch-preparation contract.

Replacement probabilities come from ``MLMConfig``. Training config resolution may replace
those raw helper defaults for the selected backbone; see the
[Config reference](../guides/config-reference.md).

## `MLMConfig`

```python
class MLMConfig(
    mlm_probability: 'float',
    mask_token_prob: 'float' = 0.8,
    random_token_prob: 'float' = 0.1,
    max_ngram: 'int' = 1,
) -> None
```

Masking configuration.

Mask selection uses DeBERTa's windowed policy: ``max_ngram=1`` selects windowed unigrams and
larger values enable whole-word n-grams. Replacement probabilities are conditional on token
selection and must sum to at most one; the remainder keeps the original token.

## `load_hf_dataset`

```python
load_hf_dataset(cfg: 'DataConfig') -> 'Any'
```

Load the configured training split through Hugging Face Datasets.

Source selection is described in [Data pipeline](../guides/data-pipeline.md#dataset-source-selection).

### Parameters

- `cfg` (`DataConfig`): Data config containing dataset source settings.

### Returns

- `Any`: Map-style Dataset or streaming IterableDataset.

## `PackedStreamingConfig`

```python
class PackedStreamingConfig(
    text_column_name: 'str',
    max_seq_length: 'int',
    seed: 'int',
    shuffle_buffer_size: 'int',
) -> None
```

Configuration for packing raw text into fixed-length token blocks.

## `PackedStreamingDataset`

```python
class PackedStreamingDataset(
    *,
    hf_dataset: 'Any',
    tokenizer: 'Any',
    cfg: 'PackedStreamingConfig',
    process_index: 'int' = 0,
    num_processes: 'int' = 1,
) -> 'None'
```

Pack streaming text into fixed-length blocks across ranks and DataLoader workers.

See [Data pipeline](../guides/data-pipeline.md#packed-streaming-path) for sample construction
and output fields.

### Members

#### `set_epoch`

```python
set_epoch(self, epoch: 'int') -> 'None'
```

`method`

Forward epoch to underlying dataset when supported.

### Parameters

- `epoch` (`int`): Training epoch.

## `SequentialStreamingDataset`

```python
class SequentialStreamingDataset(
    *,
    hf_dataset: 'Any',
    tokenizer: 'Any',
    cfg: 'PackedStreamingConfig',
    process_index: 'int' = 0,
    num_processes: 'int' = 1,
) -> 'None'
```

One-document-per-sequence dataset (reference mode without cross-document packing).
