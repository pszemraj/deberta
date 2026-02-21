"""Dataset loading helpers wrapping Hugging Face Datasets APIs."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any


def _pretty_cfg(cfg: Any) -> str:
    """Render config object for user-facing error messages.

    :param Any cfg: Config-like object.
    :return str: Best-effort readable representation.
    """
    try:
        return str(asdict(cfg))
    except Exception:
        return repr(cfg)


def load_hf_dataset(*, cfg: Any, split: str, streaming: bool) -> Any:
    """Load a dataset split using 🤗 Datasets.

    Supports:
      - load_from_disk
      - load_dataset(name)
      - load_dataset('text', data_files=...)

    :param Any cfg: Data config containing dataset source settings.
    :param str split: Split name to load.
    :param bool streaming: Whether to request a streaming dataset.
    :return Any: Map-style Dataset or streaming IterableDataset.

    If both ``dataset_name`` and ``data_files`` are set, ``data_files`` are passed through
    to ``datasets.load_dataset`` for builders that support local files (for example ``text``).
    """

    try:
        import datasets
    except Exception as e:  # pragma: no cover
        raise RuntimeError("datasets is required. Install with `pip install datasets`.") from e

    if cfg.load_from_disk:
        if streaming:
            raise ValueError(
                "--streaming true is not compatible with --load_from_disk. Set --streaming false."
            )
        ds = datasets.load_from_disk(cfg.load_from_disk)
        # Support DatasetDict
        if isinstance(ds, datasets.DatasetDict):
            if split not in ds:
                raise ValueError(
                    f"Split '{split}' not found in load_from_disk dataset. Available: {list(ds.keys())}"
                )
            ds = ds[split]
        return ds

    if cfg.dataset_name:
        load_kwargs: dict[str, Any] = {
            "split": split,
            "streaming": streaming,
            "cache_dir": cfg.cache_dir,
        }
        if cfg.dataset_config_name:
            load_kwargs["name"] = cfg.dataset_config_name
        if cfg.data_files:
            files = [p.strip() for p in cfg.data_files.split(",") if p.strip()]
            load_kwargs["data_files"] = files
        return datasets.load_dataset(
            cfg.dataset_name,
            **load_kwargs,
        )

    if cfg.data_files:
        files = [p.strip() for p in cfg.data_files.split(",") if p.strip()]
        # Use the 'text' builder; split can be 'train' etc.
        return datasets.load_dataset(
            "text",
            data_files=files,
            split=split,
            streaming=streaming,
            cache_dir=cfg.cache_dir,
        )

    raise ValueError(
        "No dataset source provided. Specify one of: --load_from_disk, --dataset_name, --data_files. "
        f"Config was: {_pretty_cfg(cfg)}"
    )
