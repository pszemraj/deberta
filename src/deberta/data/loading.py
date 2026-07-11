"""Dataset loading helpers wrapping Hugging Face Datasets APIs."""

from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Any

from deberta.config import DataConfig
from deberta.data.retry import call_with_dataset_retry

logger = logging.getLogger(__name__)


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

    source = cfg.source

    def _retry_notice(label: str) -> Any:
        """Build a labeled retry logger callback.

        :param str label: Dataset operation label.
        :return Any: Retry callback.
        """

        def _notice(attempt: int, delay: float, exc: BaseException) -> None:
            """Log one transient dataset retry.

            :param int attempt: Failed attempt number.
            :param float delay: Delay before retrying.
            :param BaseException exc: Transient failure.
            :return None: None.
            """
            logger.warning(
                "%s failed transiently (attempt %d/%d, retry in %.1fs): %s",
                label,
                attempt,
                source.retry_attempts,
                delay,
                exc,
            )

        return _notice

    if source.load_from_disk:
        if source.streaming:
            raise ValueError(
                "data.source.streaming=true is not compatible with data.source.load_from_disk. "
                "Set data.source.streaming=false."
            )
        ds = call_with_dataset_retry(
            lambda: datasets.load_from_disk(source.load_from_disk),
            attempts=source.retry_attempts,
            backoff_seconds=source.retry_backoff_seconds,
            on_retry=_retry_notice("Dataset load_from_disk"),
        )
        # Support DatasetDict
        if isinstance(ds, datasets.DatasetDict):
            if source.train_split not in ds:
                raise ValueError(
                    f"Split '{source.train_split}' not found in load_from_disk dataset. "
                    f"Available: {list(ds.keys())}"
                )
            ds = ds[source.train_split]
        return ds

    files = [p.strip() for p in source.data_files.split(",") if p.strip()] if source.data_files else []

    if source.dataset_name:
        load_kwargs: dict[str, Any] = {
            "split": source.train_split,
            "streaming": source.streaming,
        }
        if source.dataset_config_name:
            load_kwargs["name"] = source.dataset_config_name
        if files:
            load_kwargs["data_files"] = files
        return call_with_dataset_retry(
            lambda: datasets.load_dataset(source.dataset_name, **load_kwargs),
            attempts=source.retry_attempts,
            backoff_seconds=source.retry_backoff_seconds,
            on_retry=_retry_notice("Dataset load"),
        )

    if files:
        return call_with_dataset_retry(
            lambda: datasets.load_dataset(
                "text",
                data_files=files,
                split=source.train_split,
                streaming=source.streaming,
            ),
            attempts=source.retry_attempts,
            backoff_seconds=source.retry_backoff_seconds,
            on_retry=_retry_notice("Text dataset load"),
        )

    raise ValueError(
        "No dataset source provided. Set one of data.source.load_from_disk, "
        "data.source.dataset_name, or data.source.data_files. "
        f"Config was: {asdict(cfg)}"
    )
