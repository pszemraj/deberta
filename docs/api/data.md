# Data API

## Module `deberta.data`

Data utilities for DeBERTaV3 pretraining.

## `DebertaV3ElectraCollator`

```python
DebertaV3ElectraCollator(
    *,
    tokenizer: 'Any',
    cfg: 'MLMConfig',
    packed_sequences: 'bool' = False,
    block_cross_document_attention: 'bool' = True,
    emit_flash_metadata: 'bool' = True,
    pad_to_multiple_of: 'int | None' = None,
)
```

Dynamic MLM masking collator suitable for RTD/ELECTRA-style pretraining.

Produces masked ``input_ids``, MLM ``labels``, and optional attention/token-type tensors.
Packed doc-block batches carry ``doc_ids``; ``emit_flash_metadata=True`` additionally emits compact Flash routing metadata. See
[Data pipeline](../guides/data-pipeline.md#cross-document-attention-blocking) for the complete
batch-preparation contract.

Replacement probabilities come from ``MLMConfig``. Training config resolution may replace
those raw helper defaults for the selected backbone; see the [config reference](../../configs/config_reference.yaml).

## `MLMConfig`

```python
MLMConfig(
    mlm_probability: 'float',
    mask_token_prob: 'float' = 0.8,
    random_token_prob: 'float' = 0.1,
    max_ngram: 'int' = 1,
)
```

Masking configuration.

See [Collator masking behavior](../guides/data-pipeline.md#collator-masking-behavior) for mask selection and replacement semantics.

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
PackedStreamingConfig(
    text_column_name: 'str',
    max_seq_length: 'int',
    seed: 'int',
    shuffle_buffer_size: 'int',
    block_cross_document_attention: 'bool' = False,
    retry_attempts: 'int' = 3,
    retry_backoff_seconds: 'float' = 1.0,
)
```

Configuration for packing raw text into fixed-length token blocks.

The retry fields bound recovery from transient streaming failures; their attempt-count, backoff, and replay semantics are defined under `data.source.*` in the [config reference](../../configs/config_reference.yaml).

## `PackedStreamingDataset`

```python
PackedStreamingDataset(
    *,
    hf_dataset: 'Any',
    tokenizer: 'Any',
    cfg: 'PackedStreamingConfig',
    process_index: 'int' = 0,
    num_processes: 'int' = 1,
)
```

Pack source text into fixed-length blocks across ranks and DataLoader workers.

See [Data pipeline](../guides/data-pipeline.md#packed-streaming-path) for sample construction
and output fields.

The wrapper accepts HF map-style and iterable datasets. Their distinct worker-sharding behavior is documented under [dataset source selection](../guides/data-pipeline.md#dataset-source-selection).

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
SequentialStreamingDataset(
    *,
    hf_dataset: 'Any',
    tokenizer: 'Any',
    cfg: 'PackedStreamingConfig',
    process_index: 'int' = 0,
    num_processes: 'int' = 1,
)
```

One-document-per-sequence dataset (reference mode without cross-document packing).
