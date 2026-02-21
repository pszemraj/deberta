# Data pipeline

This codebase is **streaming-first** for pretraining.

## Streaming path (default)

When `data.streaming=true`, we use Hugging Face Datasets in streaming mode and wrap it in a small packing iterator:

- tokenizes each document with `add_special_tokens=False`
- appends a `[SEP]` separator **between documents**
- packs tokens into fixed-length blocks of `max_seq_length`
- emits examples shaped like:

```python
{
  "input_ids":          [CLS] + chunk + [SEP] + pad,
  "special_tokens_mask": 1 for CLS/SEP/PAD tokens, 0 otherwise,
  "attention_mask":     optional, only when padding is present
}
```

### Important detail: internal `[SEP]` tokens

Because the packer inserts `[SEP]` tokens inside `chunk` (to separate documents), we **mark those internal `[SEP]` tokens as special** in `special_tokens_mask`.

This prevents the MLM masking collator from masking them, which would otherwise degrade training.

## Non-streaming paths

If you set `data.streaming=false`, we fall back to standard `datasets.load_dataset()` or `load_from_disk()` and then perform an offline packing step that produces the same example structure as the streaming path.

## Masking

Masking is applied *dynamically* in the collator (not precomputed):

- token-level masking (fast): `train.mlm_max_ngram = 1`
- DeBERTa-style whole-word n-gram masking: `train.mlm_max_ngram > 1`

The labels follow HF convention:

- `labels[i] = original_token_id` for masked positions
- `labels[i] = -100` for all other positions

## FlashAttention and Masks

For packed pretraining batches, sequences are usually fixed-length and unpadded. In this case,
an explicit all-ones `attention_mask` is redundant and can reduce SDPA backend flexibility.

This repo keeps that path lean by:

1. Omitting `attention_mask` from packed datasets when no padding is present.
2. Having the collator try `tokenizer.pad(..., return_attention_mask=False)` when possible.
3. Dropping all-ones attention masks from the collated batch.

The model accepts `attention_mask=None` and treats it as an unpadded batch.
