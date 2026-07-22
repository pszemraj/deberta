# Training API

## Module `deberta.training`

Training entry points.

## `run_pretraining`

```python
run_pretraining(
    *,
    model_cfg: 'ModelConfig',
    data_cfg: 'DataConfig',
    train_cfg: 'TrainConfig | None' = None,
    optim_cfg: 'OptimConfig | None' = None,
    logging_cfg: 'LoggingConfig | None' = None,
    config_path: 'str | Path | None' = None,
) -> 'None'
```

Run RTD pretraining. See [Distributed training](../advanced/distributed-training.md) for the multi-GPU support boundary.

### Parameters

- `model_cfg` (`ModelConfig`): Model configuration.
- `data_cfg` (`DataConfig`): Data configuration.
- `train_cfg` (`TrainConfig | None`): Explicit training configuration, or `None` to use the selected backbone profile defaults.
- `optim_cfg` (`OptimConfig | None`): Explicit optimizer configuration, or `None` to use the selected backbone profile defaults.
- `logging_cfg` (`LoggingConfig | None`): Optional logging configuration.
- `config_path` (`str | Path | None`): Optional source config path for auto output-dir naming.

## `run_pretraining_dry_run`

```python
run_pretraining_dry_run(
    *,
    model_cfg: 'ModelConfig',
    data_cfg: 'DataConfig',
    train_cfg: 'TrainConfig | None' = None,
    optim_cfg: 'OptimConfig | None' = None,
    logging_cfg: 'LoggingConfig | None' = None,
    config_path: 'str | Path | None' = None,
) -> 'dict[str, Any]'
```

Run preflight checks for `deberta train`.

This validates configuration contracts and probes core runtime dependencies
(tokenizer, dataset access, collator output, model config construction)
without starting optimization/training loops. It may access network sources and populate
dependency caches; see the [config reference](../../configs/config_reference.yaml).

### Parameters

- `model_cfg` (`ModelConfig`): Model configuration.
- `data_cfg` (`DataConfig`): Data configuration.
- `train_cfg` (`TrainConfig | None`): Explicit training configuration, or `None` to use the selected backbone profile defaults.
- `optim_cfg` (`OptimConfig | None`): Explicit optimizer configuration, or `None` to use the selected backbone profile defaults.
- `logging_cfg` (`LoggingConfig | None`): Optional logging configuration.
- `config_path` (`str | Path | None`): Optional source config path.

### Raises

- `RuntimeError`: If a preflight stage fails.

### Returns

- `dict[str, Any]`: Summary of resolved dry-run checks.
