# FSDP2

This project targets **PyTorch 2.9.1+** and supports **FSDP2** via **🤗 Accelerate** (set `distributed_type: FSDP` + `fsdp_version: 2`).

## Provided configs

We ship two ready-to-use configs:

- `configs/fsdp2_1node.yaml`  
  Wrap class: **`DebertaRoPELayer`** (the modern RoPE+RMSNorm backbone, default)

- `configs/fsdp2_hf_deberta_1node.yaml`  
  Wrap class: **`DebertaV2Layer`** (Hugging Face DeBERTa v2/v3 compatibility path)

Key settings you generally want:

- `fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP`
- `fsdp_transformer_layer_cls_to_wrap: <your block class>`
- `fsdp_state_dict_type: SHARDED_STATE_DICT` during training (fast/robust)

This wraps **transformer blocks only**, leaving embeddings and heads in the outermost FSDP unit (important for embedding tying/sharing).

## Embedding sharing + FSDP wrapping

DeBERTaV3 RTD typically shares embeddings between generator and discriminator.

To keep FSDP stable:

- Avoid creating **shared module instances** between generator/discriminator.
- Ensure that modules that share weights do not end up in different FSDP units.

This repo implements tying via a lightweight wrapper that *references* the generator weight (weakref) rather than reusing module instances.

## Checkpoint export (SHARDED_STATE_DICT)

When you save checkpoints with `SHARDED_STATE_DICT`, gathering a full state dict inside the training process is not always reliable.

Use the dedicated exporter:

```bash
accelerate launch --config_file configs/fsdp2_1node.yaml deberta-export \
  --checkpoint_dir <RUN_DIR>/checkpoint-<STEP> \
  --output_dir <RUN_DIR>/exported_hf \
  --export_what discriminator
```

This consolidates a **FULL_STATE_DICT on rank0** and writes standalone HF backbones.

## Practical tips

- Prefer `bf16` mixed precision on modern GPUs.
- Keep `gradient_accumulation_steps` high enough to amortize communication.
- Activation checkpointing (either `--gradient_checkpointing` or accelerate config) is the easiest knob to trade compute for memory.

## Debugging

If you see errors related to:

- auto-wrap class names not found
- unexpected missing keys on load/export
- shared modules / parameter aliasing

…double-check that your accelerate config uses the correct `fsdp_transformer_layer_cls_to_wrap` for the chosen `--backbone_type`.
