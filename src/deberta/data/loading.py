"""Dataset loading helpers wrapping Hugging Face Datasets APIs."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from deberta.config import DataConfig


def load_hf_dataset(cfg: DataConfig) -> Any:
    """Load the configured training split through Hugging Face Datasets.

    Source selection is described in [Data pipeline](../guides/data-pipeline.md#dataset-source-selection).

    :param DataConfig cfg: Data config containing dataset source settings.
    :return Any: Map-style Dataset or streaming IterableDataset.
    """

    try:
        import datasets
    except Exception as e:  # pragma: no cover
        raise RuntimeError("datasets is required. Install with `pip install datasets`.") from e

    if cfg.load_from_disk:
        if cfg.streaming:
            raise ValueError(
                "--streaming true is not compatible with --load_from_disk. Set --streaming false."
            )
        ds = datasets.load_from_disk(cfg.load_from_disk)
        # Support DatasetDict
        if isinstance(ds, datasets.DatasetDict):
            if cfg.train_split not in ds:
                raise ValueError(
                    f"Split '{cfg.train_split}' not found in load_from_disk dataset. "
                    f"Available: {list(ds.keys())}"
                )
            ds = ds[cfg.train_split]
        return ds

    _files = [p.strip() for p in cfg.data_files.split(",") if p.strip()] if cfg.data_files else []

    if cfg.dataset_name:
        load_kwargs: dict[str, Any] = {
            "split": cfg.train_split,
            "streaming": cfg.streaming,
        }
        if cfg.dataset_config_name:
            load_kwargs["name"] = cfg.dataset_config_name
        if _files:
            load_kwargs["data_files"] = _files
        return datasets.load_dataset(cfg.dataset_name, **load_kwargs)

    if _files:
        return datasets.load_dataset(
            "text",
            data_files=_files,
            split=cfg.train_split,
            streaming=cfg.streaming,
        )

    raise ValueError(
        "No dataset source provided. Specify one of: --load_from_disk, --dataset_name, --data_files. "
        f"Config was: {asdict(cfg)}"
    )
