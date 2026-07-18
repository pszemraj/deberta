"""Bounded retry helpers for remote dataset operations."""

from __future__ import annotations

import errno
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")
DATASET_RETRY_ATTEMPTS = 3

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
_TRANSIENT_OS_ERRNOS = {
    errno.EAGAIN,
    errno.ECONNABORTED,
    errno.ECONNREFUSED,
    errno.ECONNRESET,
    errno.EHOSTUNREACH,
    errno.EINTR,
    errno.ENETDOWN,
    errno.ENETRESET,
    errno.ENETUNREACH,
    errno.ETIMEDOUT,
}
_TRANSIENT_HTTP_STATUS_CODES = frozenset({429, 500, 502, 503, 504})


def _http_status_code(exc: BaseException) -> int | None:
    """Return a duck-typed HTTP response status code when available.

    ``requests``/Hugging Face and ``httpx`` status exceptions both expose the
    originating response through ``exc.response``. Avoid importing either
    optional client here so dataset retry remains dependency-neutral.

    :param BaseException exc: Candidate HTTP status exception.
    :return int | None: Response status code, or None when unavailable.
    """
    response = getattr(exc, "response", None)
    raw_status = getattr(response, "status_code", None) if response is not None else None
    if raw_status is None:
        raw_status = getattr(exc, "status_code", None)
    if raw_status is None:
        return None
    try:
        return int(raw_status)
    except (TypeError, ValueError):
        return None


def is_transient_dataset_error(exc: BaseException) -> bool:
    """Return whether an exception chain represents retryable remote I/O failure.

    :param BaseException exc: Raised dataset exception.
    :return bool: True for transient network, timeout, and filesystem failures.
    """
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        status_code = _http_status_code(current)
        if status_code is not None:
            return status_code in _TRANSIENT_HTTP_STATUS_CODES
        if isinstance(current, (ConnectionError, TimeoutError)):
            return True
        if isinstance(current, OSError) and current.errno in _TRANSIENT_OS_ERRNOS:
            return True
        if type(current).__name__ in _TRANSIENT_ERROR_NAMES:
            return True
        current = current.__cause__ or current.__context__
    return False


def handle_dataset_retry_failure(
    exc: BaseException,
    *,
    attempt: int,
    on_retry: Callable[[int, BaseException], None],
) -> None:
    """Raise a terminal dataset failure or notify before retrying.

    :param BaseException exc: Failure from the current attempt.
    :param int attempt: One-based current attempt.
    :param Callable[[int, BaseException], None] on_retry: Retry notification callback.
    :raises BaseException: If the failure is non-transient or the budget is exhausted.
    """

    if int(attempt) >= DATASET_RETRY_ATTEMPTS or not is_transient_dataset_error(exc):
        raise exc
    on_retry(int(attempt), exc)


def call_with_dataset_retry(
    operation: Callable[[], T],
    *,
    on_retry: Callable[[int, BaseException], None],
) -> T:
    """Run one dataset operation with bounded transient retry.

    :param Callable[[], T] operation: Operation to execute.
    :param Callable[[int, BaseException], None] on_retry: Retry notification callback.
    :return T: Operation result.
    """
    for attempt in range(1, DATASET_RETRY_ATTEMPTS + 1):
        try:
            return operation()
        except Exception as exc:
            handle_dataset_retry_failure(
                exc,
                attempt=attempt,
                on_retry=on_retry,
            )
