# Modeling API

## Module `deberta.modeling`

Model components for DeBERTaV3 RTD pretraining.

## `DebertaV3RTDPretrainer`

```python
class DebertaV3RTDPretrainer(
    *,
    discriminator_backbone: 'nn.Module',
    generator_backbone: 'nn.Module',
    disc_config: 'Any',
    gen_config: 'Any',
    embedding_sharing: 'str' = 'gdes',
    additional_forbidden_token_ids: 'Iterable[int] | None' = None,
) -> 'None'
```

Generator + discriminator pretraining module (DeBERTaV3 / ELECTRA objective).

### Members

#### `sync_discriminator_embeddings_from_generator`

```python
sync_discriminator_embeddings_from_generator(self) -> 'None'
```

`method`

Sync GDES base weights from generator embedding weights.

Must be called after each optimizer step and after checkpoint load.

#### `forward_generator_phase`

```python
forward_generator_phase(
    self,
    *,
    input_ids: 'torch.Tensor',
    attention_mask: 'torch.Tensor | None' = None,
    labels: 'torch.Tensor',
    token_type_ids: 'torch.Tensor | None' = None,
    position_ids: 'torch.Tensor | None' = None,
    sampling_temperature: 'float' = 1.0,
    flash_meta: 'FlashBatchMeta | None' = None,
) -> 'RTDGeneratorPhaseOutput'
```

`method`

Run generator forward/corruption only, returning discriminator targets.

### Parameters

- `input_ids` (`torch.Tensor`): Masked input ids.
- `attention_mask` (`torch.Tensor | None`): Optional attention mask.
- `labels` (`torch.Tensor`): MLM labels with ``-100`` ignore index.
- `token_type_ids` (`torch.Tensor | None`): Optional token type ids.
- `position_ids` (`torch.Tensor | None`): Optional document-local position ids.
- `sampling_temperature` (`float`): Generator sampling temperature.
- `flash_meta` (`FlashBatchMeta | None`): Optional FlashDeBERTa metadata bundle.

### Returns

- `RTDGeneratorPhaseOutput`: Generator loss and corruption artifacts.

#### `forward_discriminator_phase`

```python
forward_discriminator_phase(
    self,
    *,
    input_ids: 'torch.Tensor',
    corrupted_input_ids: 'torch.Tensor',
    disc_labels: 'torch.Tensor',
    attention_mask: 'torch.Tensor | None' = None,
    token_type_ids: 'torch.Tensor | None' = None,
    position_ids: 'torch.Tensor | None' = None,
    doc_context_index: 'torch.Tensor | None' = None,
    flash_meta: 'FlashBatchMeta | None' = None,
) -> 'RTDDiscriminatorPhaseOutput'
```

`method`

Run discriminator scoring only, given prebuilt corrupted ids/labels.

### Parameters

- `input_ids` (`torch.Tensor`): Input ids used only for active-token masking.
- `corrupted_input_ids` (`torch.Tensor`): Corrupted ids sampled from generator logits.
- `disc_labels` (`torch.Tensor`): Binary RTD labels.
- `attention_mask` (`torch.Tensor | None`): Optional attention mask.
- `token_type_ids` (`torch.Tensor | None`): Optional token type ids.
- `position_ids` (`torch.Tensor | None`): Optional document-local position ids.
- `doc_context_index` (`torch.Tensor | None`): Optional CLS index per token.
- `flash_meta` (`FlashBatchMeta | None`): Optional FlashDeBERTa metadata bundle.

### Returns

- `RTDDiscriminatorPhaseOutput`: Discriminator loss and metrics.

#### `forward`

```python
forward(
    self,
    *,
    input_ids: 'torch.Tensor',
    attention_mask: 'torch.Tensor | None' = None,
    labels: 'torch.Tensor | None' = None,
    token_type_ids: 'torch.Tensor | None' = None,
    position_ids: 'torch.Tensor | None' = None,
    doc_context_index: 'torch.Tensor | None' = None,
    sampling_temperature: 'float' = 1.0,
    gen_loss_weight: 'float' = 1.0,
    disc_loss_weight: 'float' = 50.0,
    phase: 'str' = 'both',
    corrupted_input_ids: 'torch.Tensor | None' = None,
    disc_labels: 'torch.Tensor | None' = None,
    flash_meta: 'FlashBatchMeta | None' = None,
) -> 'RTDOutput | RTDGeneratorPhaseOutput | RTDDiscriminatorPhaseOutput'
```

`method`

Run RTD forward in combined or phase-specific mode.

### Parameters

- `input_ids` (`torch.Tensor`): Masked input ids.
- `attention_mask` (`torch.Tensor | None`): Optional attention mask.
- `labels` (`torch.Tensor | None`): MLM labels with ``-100`` ignore index.
- `token_type_ids` (`torch.Tensor | None`): Optional token type ids.
- `position_ids` (`torch.Tensor | None`): Optional document-local position ids.
- `doc_context_index` (`torch.Tensor | None`): Optional CLS index per token.
- `sampling_temperature` (`float`): Generator sampling temperature.
- `gen_loss_weight` (`float`): Generator loss weight.
- `disc_loss_weight` (`float`): Discriminator loss weight.
- `phase` (`str`): One of ``both|generator|discriminator``.
- `corrupted_input_ids` (`torch.Tensor | None`): Precomputed corrupted ids for ``phase='discriminator'``.
- `disc_labels` (`torch.Tensor | None`): Precomputed RTD labels for ``phase='discriminator'``.
- `flash_meta` (`FlashBatchMeta | None`): Optional FlashDeBERTa metadata bundle.

### Returns

- `RTDOutput | RTDGeneratorPhaseOutput | RTDDiscriminatorPhaseOutput`: Combined output for ``phase='both'``; phase-local outputs otherwise.

## `build_backbone_configs`

```python
build_backbone_configs(
    *,
    model_cfg: 'ModelConfig',
    tokenizer: 'Any',
    max_position_embeddings: 'int',
) -> 'tuple[Any, Any]'
```

Build discriminator + generator configs.

- For backbone_type='hf_deberta_v2': discriminator config is synthesized from repo defaults
for scratch runs, or loaded from ``model.pretrained.discriminator_path`` for pretrained runs.
Generator config is derived from discriminator unless ``model.pretrained.generator_path``
is set, in which case that config is loaded.
- For backbone_type='rope': returns DebertaRoPEConfig instances.

### Parameters

- `model_cfg` (`ModelConfig`): User model configuration.
- `tokenizer` (`Any`): Tokenizer used for vocab/pad metadata.
- `max_position_embeddings` (`int`): Sequence length budget.

### Returns

- `tuple[Any, Any]`: Discriminator and generator configs.

## `build_backbones`

```python
build_backbones(
    *,
    model_cfg: 'ModelConfig',
    disc_config: 'Any',
    gen_config: 'Any',
    load_pretrained_weights: 'bool' = True,
) -> 'tuple[Any, Any]'
```

Instantiate discriminator + generator backbones.

### Parameters

- `model_cfg` (`ModelConfig`): User model configuration.
- `disc_config` (`Any`): Discriminator config object.
- `gen_config` (`Any`): Generator config object.
- `load_pretrained_weights` (`bool`): Whether to load pretrained weights from resolved model sources when ``model.from_scratch=false``. Set ``False`` for resume/export flows that will load from an accelerate checkpoint immediately after instantiation.

### Returns

- `tuple[Any, Any]`: Instantiated discriminator and generator modules.

## `DebertaV2Config`

```python
class DebertaV2Config(
    vocab_size=128100,
    hidden_size=1536,
    num_hidden_layers=24,
    num_attention_heads=24,
    intermediate_size=6144,
    hidden_act='gelu',
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1,
    max_position_embeddings=512,
    type_vocab_size=0,
    initializer_range=0.02,
    layer_norm_eps=1e-07,
    relative_attention=False,
    max_relative_positions=-1,
    pad_token_id=0,
    position_biased_input=True,
    pos_att_type=None,
    pooler_dropout=0,
    pooler_hidden_act='gelu',
    legacy=True,
    **kwargs,
)
```

Re-export of `transformers.models.deberta_v2.configuration_deberta_v2.DebertaV2Config`.

In this package it configures the repo-native `DebertaV2Model`.

## `DebertaV2Model`

```python
class DebertaV2Model(config: 'DebertaV2Config') -> 'None'
```

Native encoder-only DeBERTa-v2 model returning ``BaseModelOutput``.

### Members

#### `get_input_embeddings`

```python
get_input_embeddings(self) -> 'nn.Module'
```

`method`

Return word embedding module.

### Returns

- `nn.Module`: Input embedding module.

#### `set_input_embeddings`

```python
set_input_embeddings(self, new_embeddings: 'nn.Module') -> 'None'
```

`method`

Replace input word embedding module.

### Parameters

- `new_embeddings` (`nn.Module`): Replacement embedding module.

#### `forward`

```python
forward(
    self,
    input_ids: 'torch.Tensor | None' = None,
    attention_mask: 'torch.Tensor | None' = None,
    token_type_ids: 'torch.Tensor | None' = None,
    position_ids: 'torch.Tensor | None' = None,
    inputs_embeds: 'torch.Tensor | None' = None,
    output_attentions: 'bool | None' = None,
    output_hidden_states: 'bool | None' = None,
    return_dict: 'bool | None' = None,
    flash_meta: 'FlashBatchMeta | None' = None,
) -> 'BaseModelOutput | tuple[torch.Tensor, ...]'
```

`method`

Run DeBERTa-v2 encoder forward pass.

### Parameters

- `input_ids` (`torch.Tensor | None`): Optional input token ids.
- `attention_mask` (`torch.Tensor | None`): Optional attention mask.
- `token_type_ids` (`torch.Tensor | None`): Optional token type ids.
- `position_ids` (`torch.Tensor | None`): Optional position ids.
- `inputs_embeds` (`torch.Tensor | None`): Optional precomputed embeddings.
- `output_attentions` (`bool | None`): Optional attention-output flag.
- `output_hidden_states` (`bool | None`): Optional hidden-state-output flag.
- `return_dict` (`bool | None`): Optional return-format flag.
- `flash_meta` (`FlashBatchMeta | None`): Optional FlashDeBERTa metadata bundle.

### Raises

- `ValueError`: If both/neither ``input_ids`` and ``inputs_embeds`` are set.

### Returns

- `BaseModelOutput | tuple[torch.Tensor, ...]`: Model outputs.

## `DebertaRoPEConfig`

```python
class DebertaRoPEConfig(
    *,
    vocab_size: 'int' = 50265,
    hidden_size: 'int' = 768,
    num_hidden_layers: 'int' = 12,
    num_attention_heads: 'int' = 12,
    intermediate_size: 'int' = 3072,
    hidden_act: 'str' = 'gelu',
    ffn_type: 'str' = 'swiglu',
    use_bias: 'bool' = False,
    hidden_dropout_prob: 'float' = 0.0,
    attention_probs_dropout_prob: 'float' = 0.0,
    max_position_embeddings: 'int' = 512,
    type_vocab_size: 'int' = 2,
    pad_token_id: 'int' = 0,
    rope_theta: 'float' = 10000.0,
    rotary_pct: 'float' = 1.0,
    use_absolute_position_embeddings: 'bool' = False,
    norm_eps: 'float' = 1e-06,
    norm_arch: 'str' = 'post',
    keel_alpha_init: 'float | None' = None,
    keel_alpha_learnable: 'bool' = False,
    attention_implementation: 'str' = 'sdpa',
    initializer_range: 'float' = 0.02,
    **kwargs: 'Any',
) -> 'None'
```

Config for the modernized RoPE encoder backbone.

## `DebertaRoPEModel`

```python
class DebertaRoPEModel(config: 'DebertaRoPEConfig') -> 'None'
```

Encoder-only model with HF-style ``BaseModelOutput`` support.

### Members

#### `get_input_embeddings`

```python
get_input_embeddings(self) -> 'nn.Module'
```

`method`

Return input word embedding module.

### Returns

- `nn.Module`: Word embedding module.

#### `set_input_embeddings`

```python
set_input_embeddings(self, value: 'nn.Module') -> 'None'
```

`method`

Replace input word embedding module.

### Parameters

- `value` (`nn.Module`): New word embedding module.

#### `forward`

```python
forward(
    self,
    input_ids: 'torch.Tensor',
    attention_mask: 'torch.Tensor | None' = None,
    token_type_ids: 'torch.Tensor | None' = None,
    position_ids: 'torch.Tensor | None' = None,
    output_hidden_states: 'bool | None' = None,
    output_attentions: 'bool | None' = None,
    return_dict: 'bool | None' = None,
) -> 'BaseModelOutput | tuple[torch.Tensor, ...]'
```

`method`

Run encoder forward pass.

### Parameters

- `input_ids` (`torch.Tensor`): Input token ids.
- `attention_mask` (`torch.Tensor | None`): Optional attention mask. ``None`` means unpadded input (fast path); callers must pass a mask when padding exists.
- `token_type_ids` (`torch.Tensor | None`): Optional segment ids.
- `position_ids` (`torch.Tensor | None`): Optional learned absolute-position ids. Packed document rows use local ids; rotary attention itself is invariant to each isolated segment's constant row offset.
- `output_hidden_states` (`bool | None`): Optional hidden-state output flag.
- `output_attentions` (`bool | None`): Optional attention output flag.
- `return_dict` (`bool | None`): Optional return-dataclass flag.

### Returns

- `BaseModelOutput | tuple[torch.Tensor, ...]`: Model output container/tuple.

## `DebertaRoPELayer`

```python
class DebertaRoPELayer(config: 'DebertaRoPEConfig', *, alpha_init: 'float') -> 'None'
```

One encoder layer with attention + MLP.

Norm placement is controlled by config.norm_arch:
- 'post' : classic Post-Norm (RMSNorm after each residual addition)
- 'keel' : KEEL-style: RMSNorm(alpha*x + F(RMSNorm(x))) per sub-layer

### Members

#### `forward`

```python
forward(self, x: 'torch.Tensor', attention_mask: 'torch.Tensor | None') -> 'torch.Tensor'
```

`method`

Run one encoder layer pass.

### Parameters

- `x` (`torch.Tensor`): Input hidden states.
- `attention_mask` (`torch.Tensor | None`): Binary attention mask.

### Returns

- `torch.Tensor`: Output hidden states.
