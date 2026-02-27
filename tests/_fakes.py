"""Shared test stubs for accelerator, tokenizer, and W&B fakes.

Import these instead of redefining inline in each test module.
Subclass or set instance attributes for test-specific variations.
"""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch


class DummyTokenizer:
    """Minimal tokenizer stub for unit tests (no network/model downloads)."""

    def __init__(self, vocab_size: int = 128) -> None:
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.cls_token_id = 1
        self.sep_token_id = 2
        self.mask_token_id = 3
        self.all_special_ids = [self.pad_token_id, self.cls_token_id, self.sep_token_id, self.mask_token_id]
        self.all_special_tokens = ["[PAD]", "[CLS]", "[SEP]", "[MASK]"]
        self._id_to_tok = {
            self.pad_token_id: "[PAD]",
            self.cls_token_id: "[CLS]",
            self.sep_token_id: "[SEP]",
            self.mask_token_id: "[MASK]",
        }

    def __len__(self) -> int:
        return self.vocab_size

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
        return_attention_mask: bool = False,
        return_token_type_ids: bool = False,
    ) -> dict[str, list[int]]:
        del add_special_tokens, return_attention_mask, return_token_type_ids
        ids: list[int] = []
        for w in text.strip().split():
            ids.append(10 + (abs(hash(w)) % (self.vocab_size - 10)))
        return {"input_ids": ids}

    def convert_ids_to_tokens(self, ids: list[int]) -> list[str]:
        out: list[str] = []
        for i in ids:
            if i in self._id_to_tok:
                out.append(self._id_to_tok[i])
            else:
                out.append("▁" + str(i))
        return out

    def get_special_tokens_mask(
        self, token_ids_0: list[int], already_has_special_tokens: bool = True
    ) -> list[int]:
        del already_has_special_tokens
        specials = set(self.all_special_ids)
        return [1 if int(tid) in specials else 0 for tid in token_ids_0]

    def pad(
        self,
        features: list[dict[str, Any]],
        return_tensors: str = "pt",
        pad_to_multiple_of: int | None = None,
        return_attention_mask: bool = True,
    ):
        max_len = max(len(f["input_ids"]) for f in features)
        if pad_to_multiple_of is not None:
            if max_len % pad_to_multiple_of != 0:
                max_len = ((max_len // pad_to_multiple_of) + 1) * pad_to_multiple_of

        batch: dict[str, list[list[int]]] = {}
        for k in features[0].keys():
            batch[k] = []
        for f in features:
            for k, v in f.items():
                if isinstance(v, list):
                    pad_val = 0
                    if k == "special_tokens_mask":
                        pad_val = 1
                    batch[k].append(v + [pad_val] * (max_len - len(v)))
                else:
                    raise TypeError(f"Unsupported feature type for {k}: {type(v)}")

        if "attention_mask" not in batch and return_attention_mask:
            batch["attention_mask"] = [
                [0 if tid == self.pad_token_id else 1 for tid in row] for row in batch["input_ids"]
            ]

        if return_tensors == "pt":
            return {k: torch.tensor(v, dtype=torch.long) for k, v in batch.items()}
        return batch

    def save_pretrained(self, path: str) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)


class FakeWandbRun:
    """Stub for ``wandb.sdk.wandb_run.Run`` with tracking attributes."""

    def __init__(self) -> None:
        self.summary: dict[str, Any] = {}
        self.logged: list[tuple[dict[str, Any], int | None]] = []
        self.watch_calls: list[tuple[torch.nn.Module, dict[str, Any]]] = []
        self.saved_paths: list[Path] = []
        self.finished_exit_code: int | None = None

    def log(self, row: dict[str, Any], step: int | None = None) -> None:
        self.logged.append((dict(row), step))

    def watch(self, model: torch.nn.Module, **kwargs: Any) -> None:
        self.watch_calls.append((model, dict(kwargs)))

    def save(self, path: str, **kwargs: Any) -> None:
        del kwargs
        self.saved_paths.append(Path(path))

    def finish(self, exit_code: int = 0) -> None:
        self.finished_exit_code = int(exit_code)


class FakeAccelerator:
    """Stub for ``accelerate.Accelerator`` covering the full training-loop interface.

    Accepts arbitrary ``**kwargs`` in ``__init__`` so tests can pass
    ``gradient_accumulation_steps``, ``log_with``, ``mixed_precision``, etc.
    without updating this class.
    """

    def __init__(self, **kwargs: Any) -> None:
        del kwargs
        self.is_main_process = True
        self.process_index = 0
        self.num_processes = 1
        self.device = torch.device("cpu")
        self.state = "fake-accelerator"
        self.logged_rows: list[tuple[dict[str, Any], int | None]] = []
        self.tracker_init_calls: list[dict[str, Any]] = []
        self.ended = False
        self.wandb_run = FakeWandbRun()

    def wait_for_everyone(self) -> None:
        return None

    def prepare(self, *objs: Any) -> Any:
        return objs

    def unwrap_model(self, model: Any, **kwargs: Any) -> Any:
        del kwargs
        return model

    def no_sync(self, model: Any) -> Any:
        del model
        return nullcontext()

    def backward(self, loss: torch.Tensor) -> None:
        del loss

    def clip_grad_norm_(self, params: Any, max_norm: float) -> None:
        del params, max_norm

    def reduce(self, tensor: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
        del reduction
        return tensor

    def gather(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor

    def init_trackers(
        self,
        project_name: str,
        config: dict[str, Any] | None = None,
        init_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.tracker_init_calls.append(
            {
                "project_name": project_name,
                "config": dict(config or {}),
                "init_kwargs": dict(init_kwargs or {}),
            }
        )

    def get_tracker(self, name: str, unwrap: bool = True) -> Any:
        del unwrap
        if name != "wandb":
            raise KeyError(name)
        return self.wandb_run

    def log(self, row: dict[str, Any], step: int | None = None) -> None:
        self.logged_rows.append((dict(row), step))

    def load_state(self, ckpt: str, **kwargs: Any) -> None:
        del ckpt, kwargs

    def save_state(self, output_dir: str | None = None) -> None:
        del output_dir

    def get_state_dict(self, model: Any, unwrap: bool = True) -> dict[str, Any]:
        del model, unwrap
        return {}

    def end_training(self) -> None:
        self.ended = True
