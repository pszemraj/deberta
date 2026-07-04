"""FlashDeBERTa dependency version guard."""

from __future__ import annotations

from importlib import metadata

REQUIRED_FLASHDEBERTA_VERSION = "0.0.7"


def flashdeberta_distribution_version() -> str | None:
    """Return the installed FlashDeBERTa distribution version.

    :return str | None: Installed package version, or ``None`` when unavailable.
    """

    try:
        return metadata.version("flashdeberta")
    except metadata.PackageNotFoundError:
        return None


def flashdeberta_version_error() -> Exception | None:
    """Return a version mismatch error for the supported FlashDeBERTa pin.

    :return Exception | None: RuntimeError on missing/mismatched version, else ``None``.
    """

    installed = flashdeberta_distribution_version()
    if installed == REQUIRED_FLASHDEBERTA_VERSION:
        return None
    if installed is None:
        return RuntimeError(
            "flashdeberta distribution metadata was not found; install the pinned optional extra "
            f"`.[flash]` with flashdeberta=={REQUIRED_FLASHDEBERTA_VERSION}."
        )
    return RuntimeError(
        "Unsupported flashdeberta version "
        f"{installed!r}; this branch requires flashdeberta=={REQUIRED_FLASHDEBERTA_VERSION}."
    )


def require_flashdeberta_version() -> str:
    """Return the installed version or raise on mismatch.

    :raises RuntimeError: If FlashDeBERTa is missing or not at the supported pin.
    :return str: Installed package version.
    """

    error = flashdeberta_version_error()
    if error is not None:
        raise error
    return REQUIRED_FLASHDEBERTA_VERSION


__all__ = [
    "REQUIRED_FLASHDEBERTA_VERSION",
    "flashdeberta_distribution_version",
    "flashdeberta_version_error",
    "require_flashdeberta_version",
]
