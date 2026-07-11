"""Bounded retry helpers for remote dataset operations."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")

_TRANSIENT_ERROR_NAMES = {
    "ChunkedEncodingError",
    "ConnectError",
    "ConnectionError",
    "ConnectionResetError",
    "IncompleteRead",
    "ReadTimeout",
    "ReadTimeoutError",
    "RemoteDisconnected",
    "Timeout",
    "TimeoutError",
}


def is_transient_dataset_error(exc: BaseException) -> bool:
    """Return whether an exception chain represents retryable remote I/O failure.

    :param BaseException exc: Raised dataset exception.
    :return bool: True for transient network, timeout, and filesystem failures.
    """
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, (ConnectionError, OSError, TimeoutError)):
            return True
        if type(current).__name__ in _TRANSIENT_ERROR_NAMES:
            return True
        current = current.__cause__ or current.__context__
    return False


def retry_delay_seconds(*, backoff_seconds: float, failed_attempt: int) -> float:
    """Return capped exponential retry delay.

    :param float backoff_seconds: Initial delay.
    :param int failed_attempt: One-based failed-attempt number.
    :return float: Delay capped at 60 seconds.
    """
    return min(max(0.0, float(backoff_seconds)) * (2 ** max(0, int(failed_attempt) - 1)), 60.0)


def handle_dataset_retry_failure(
    exc: BaseException,
    *,
    attempt: int,
    attempts: int,
    backoff_seconds: float,
    on_retry: Callable[[int, float, BaseException], None],
) -> None:
    """Raise a terminal dataset failure or notify and wait before retrying.

    :param BaseException exc: Failure from the current attempt.
    :param int attempt: One-based current attempt.
    :param int attempts: Total attempt budget.
    :param float backoff_seconds: Initial exponential-backoff delay.
    :param Callable[[int, float, BaseException], None] on_retry: Retry notification callback.
    :raises BaseException: If the failure is non-transient or the budget is exhausted.
    """

    limit = max(1, int(attempts))
    if int(attempt) >= limit or not is_transient_dataset_error(exc):
        raise exc
    delay = retry_delay_seconds(backoff_seconds=backoff_seconds, failed_attempt=attempt)
    on_retry(int(attempt), delay, exc)
    if delay > 0.0:
        time.sleep(delay)


def call_with_dataset_retry(
    operation: Callable[[], T],
    *,
    attempts: int,
    backoff_seconds: float,
    on_retry: Callable[[int, float, BaseException], None],
) -> T:
    """Run one dataset operation with bounded transient retry.

    :param Callable[[], T] operation: Operation to execute.
    :param int attempts: Total attempts, including the first.
    :param float backoff_seconds: Initial exponential-backoff delay.
    :param Callable[[int, float, BaseException], None] on_retry: Retry notification callback.
    :return T: Operation result.
    """
    limit = max(1, int(attempts))
    for attempt in range(1, limit + 1):
        try:
            return operation()
        except Exception as exc:
            handle_dataset_retry_failure(
                exc,
                attempt=attempt,
                attempts=limit,
                backoff_seconds=backoff_seconds,
                on_retry=on_retry,
            )
    raise RuntimeError("unreachable retry state")
