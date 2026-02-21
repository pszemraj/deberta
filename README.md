# DeBERTaV3 Modernized Pretraining (RTD / ELECTRA-style)

A **PyTorch 2.9.1+** pretraining toolkit for DeBERTaV3-style **Replaced Token Detection (RTD)** with a modern encoder backbone and first-class **Accelerate + FSDP2** support.

This **v3.1** update includes:

- **RoPE** replacing DeBERTa’s *disentangled position bias*  
  → compatible with **PyTorch SDPA / FlashAttention** kernels and better length generalization
- **Post-Norm topology** retained (encoder-depth friendly), swapping **LayerNorm → RMSNorm**
- Optional **KEEL** residual topology via config (`norm_arch: keel`) for extra stability margin
- **Streaming-first** data pipeline (default), plus `load_from_disk()` and non-streaming `load_dataset()`
- Optional **DeBERTa-style whole-word n-gram masking** (`mlm_max_ngram > 1`)
- **FSDP2-safe embedding sharing** (`none | es | gdes`) + a dedicated exporter that consolidates sharded checkpoints

---

## Install

```bash
python -m venv .venv
source .venv/bin/activate

pip install -U pip
pip install -e .
```

Optional extras:

```bash
pip install -e '.[wandb]'
```

---

## Quickstart

### 1) YAML config (recommended)

```bash
accelerate launch --config_file configs/fsdp2_1node.yaml \
  deberta-pretrain configs/pretrain_rope_c4_en.yaml
```

The YAML config format supports either:

- nested sections (`model:`, `data:`, `train:`) — recommended
- a flat dict with keys from `ModelConfig`, `DataConfig`, `TrainConfig`

See `configs/` for examples.

### 2) CLI flags

Example: stream C4 (English) and train a RoPE+RMSNorm Post-Norm encoder.

```bash
accelerate launch --config_file configs/fsdp2_1node.yaml deberta-pretrain \
  --dataset_name c4 --dataset_config_name en --train_split train --streaming true \
  --tokenizer_name_or_path microsoft/deberta-v3-base \
  --backbone_type rope --from_scratch true \
  --norm_arch post --rotary_pct 1.0 --rope_theta 10000 \
  --max_seq_length 512 \
  --per_device_train_batch_size 2 --gradient_accumulation_steps 16 \
  --learning_rate 5e-4 --max_steps 10000 \
  --disc_loss_weight 50.0 \
  --output_dir runs/deberta_rope_rtd
```

Notes:
- Attention uses **`torch.nn.functional.scaled_dot_product_attention`** (`--attention_implementation sdpa`) and will dispatch to FlashAttention kernels when available.
- For length generalization, keep `--use_absolute_position_embeddings false` (default).

---

## Masking options

- **Token-level masking (fast, default):** `mlm_max_ngram = 1`
- **Whole-word n-gram masking (closer to DeBERTa):** set `mlm_max_ngram = 3` (or similar)

Replacement probabilities:
- `mask_token_prob` (default 0.8)
- `random_token_prob` (default 0.1)
- remaining probability keeps the original token

---

## KEEL mode (optional)

KEEL is an optional topology you can toggle for extra margin at ~28 layers without switching to Pre-LN:

```bash
accelerate launch --config_file configs/fsdp2_1node.yaml deberta-pretrain \
  configs/pretrain_rope_c4_en.yaml \
  # and in the YAML: model.norm_arch: keel
```

See `docs/keel-paper-technical-overview.md` for a short technical overview and rationale.

---

## Legacy HF DeBERTa path (optional)

If you want to train using the Hugging Face `DebertaV2Model` implementation (original disentangled attention), use:

- `--backbone_type hf_deberta_v2`
- `configs/fsdp2_hf_deberta_1node.yaml` (wrap class `DebertaV2Layer`)

```bash
accelerate launch --config_file configs/fsdp2_hf_deberta_1node.yaml deberta-pretrain \
  --dataset_name c4 --dataset_config_name en --train_split train --streaming true \
  --tokenizer_name_or_path microsoft/deberta-v3-base \
  --backbone_type hf_deberta_v2 --from_scratch true \
  --max_seq_length 512 --max_steps 10000 \
  --output_dir runs/deberta_hf_deberta_rtd
```

**Important:** RoPE/RMSNorm/KEEL are **not** applied in `hf_deberta_v2` mode.

---

## Exporting from FSDP2 sharded checkpoints

Training attempts a best-effort export to `output_dir/final_hf/`, but for **FSDP2 + SHARDED_STATE_DICT** you should use the dedicated exporter.

### Guaranteed consolidation + export (`deberta-export`)

Run with the **same Accelerate config** you trained with:

```bash
accelerate launch --config_file configs/fsdp2_1node.yaml -m deberta.export_cli \
  --checkpoint_dir runs/deberta_rope_rtd/checkpoint-10000 \
  --export_what discriminator \
  --output_dir runs/deberta_rope_rtd/exported_hf
```

This will:
1) load the sharded checkpoint,
2) gather a **FULL_STATE_DICT on rank0**,
3) merge tied embeddings (`es` / `gdes`) into standalone HF backbones,
4) write `exported_hf/discriminator/` (and tokenizer at `exported_hf/`).

---

## Repo layout

- `src/deberta/` – core package
- `src/deberta/training/pretrain.py` – training loop
- `src/deberta/export_cli.py` – consolidation + export tool
- `configs/` – accelerate configs (FSDP2 examples + YAML training configs)
- `docs/` – minimal docs for key pieces

---

## Next steps (not implemented yet)

Planned follow-ups (per discussion):
- SwiGLU FFN
- Longer context (2048–4096)
- FlashAttention-specific tuning
