#!/usr/bin/env python3
"""Generate static Markdown API docs from package docstrings."""

from __future__ import annotations

import argparse
import importlib
import inspect
import re
import warnings
from pathlib import Path
from types import ModuleType
from typing import Any

DEFAULT_MODULE_MAP: dict[str, str] = {
    "modeling": "deberta.modeling",
    "data": "deberta.data",
    "training": "deberta.training",
}

warnings.filterwarnings(
    "ignore",
    message="CUDA initialization: Unexpected error from cudaGetDeviceCount.*",
    category=UserWarning,
)


def _format_docstring(value: Any) -> str:
    doc = inspect.getdoc(value)
    if not doc:
        return "No docstring available."
    return _render_docstring_markdown(doc.strip())


def _format_signature(value: Any, *, prefix: str = "") -> str | None:
    try:
        sig = inspect.signature(value)
    except (TypeError, ValueError):
        return None
    return f"{prefix}{sig}" if prefix else f"{sig}"


def _render_signature_block(signature: str) -> list[str]:
    return ["```python", signature, "```", ""]


def _clean_doc_lines(doc: str) -> list[str]:
    return [line.rstrip() for line in doc.splitlines()]


def _consume_indented_continuation(lines: list[str], start: int) -> tuple[list[str], int]:
    """Consume indented continuation lines for a docstring field."""
    extras: list[str] = []
    i = start
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
            continue
        if line.lstrip().startswith(":"):
            break
        # Continuation lines in dedented docstrings still carry leading spaces.
        if len(line) > len(line.lstrip()):
            extras.append(line.strip())
            i += 1
            continue
        break
    return extras, i


def _join_fragments(parts: list[str]) -> str:
    return " ".join(part for part in parts if part).strip()


def _parse_hf_arguments_block(lines: list[str], start: int) -> tuple[list[str], int]:
    """Parse Hugging Face-style ``Arguments:`` docstring blocks into Markdown bullets."""
    entries: list[tuple[str, str, str]] = []
    i = start + 1
    arg_re = re.compile(r"^\s{4}([A-Za-z_][A-Za-z0-9_]*)\s*\((.*?)\)\s*:\s*(.*)$")
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
            continue
        if not line.startswith("    "):
            break

        m = arg_re.match(line)
        if m is None:
            i += 1
            continue

        name, meta, desc = m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
        desc_parts = [desc] if desc else []
        i += 1
        while i < len(lines):
            nxt = lines[i]
            if not nxt.strip():
                i += 1
                continue
            if arg_re.match(nxt):
                break
            if nxt.startswith("        ") or nxt.startswith("    "):
                desc_parts.append(nxt.strip())
                i += 1
                continue
            break
        entries.append((name, meta, _join_fragments(desc_parts)))

    rendered: list[str] = ["### Arguments", ""]
    for name, meta, desc in entries:
        bullet = f"- `{name}` ({meta})"
        if desc:
            bullet += f": {desc}"
        rendered.append(bullet)
    rendered.append("")
    return rendered, i


def _parse_sphinx_param(line: str) -> tuple[str, str | None, str]:
    """Parse ``:param`` lines supporting typed and untyped forms."""
    m = re.match(r"^:param\s+(.+?)\s*:\s*(.*)$", line.strip())
    if m is None:
        raise ValueError(f"Invalid :param line: {line!r}")
    lhs = m.group(1).strip()
    desc = m.group(2).strip()
    parts = lhs.rsplit(" ", maxsplit=1)
    if len(parts) == 2:
        type_part, name = parts[0].strip(), parts[1].strip()
        if name:
            return name, type_part or None, desc
    return lhs, None, desc


def _parse_sphinx_raises(line: str) -> tuple[str, str]:
    m = re.match(r"^:raises\s+(.+?)\s*:\s*(.*)$", line.strip())
    if m is None:
        raise ValueError(f"Invalid :raises line: {line!r}")
    return m.group(1).strip(), m.group(2).strip()


def _parse_sphinx_return(line: str) -> tuple[str | None, str]:
    stripped = line.strip()
    m_typed = re.match(r"^:return\s+(.+?)\s*:\s*(.*)$", stripped)
    if m_typed is not None:
        return m_typed.group(1).strip(), m_typed.group(2).strip()
    m_untyped = re.match(r"^:return:\s*(.*)$", stripped)
    if m_untyped is not None:
        return None, m_untyped.group(1).strip()
    raise ValueError(f"Invalid :return line: {line!r}")


def _render_docstring_markdown(doc: str) -> str:
    """Convert docstrings into GitHub-friendly Markdown blocks."""
    lines = _clean_doc_lines(doc)
    out: list[str] = []
    current_field_section: str | None = None

    def _ensure_field_section(section: str) -> None:
        nonlocal current_field_section
        if current_field_section == section:
            return
        if out and out[-1] != "":
            out.append("")
        heading = {
            "params": "### Parameters",
            "returns": "### Returns",
            "raises": "### Raises",
        }[section]
        out.extend([heading, ""])
        current_field_section = section

    i = 0
    while i < len(lines):
        raw = lines[i]
        stripped = raw.strip()

        if stripped == "Arguments:":
            block, i = _parse_hf_arguments_block(lines, i)
            if out and out[-1] != "":
                out.append("")
            out.extend(block)
            current_field_section = None
            continue

        if stripped in {"Example:", "Examples:"}:
            if out and out[-1] != "":
                out.append("")
            out.extend([f"### {stripped[:-1]}", ""])
            current_field_section = None
            i += 1
            continue

        if stripped.startswith(":param "):
            name, param_type, desc = _parse_sphinx_param(stripped)
            continuation, next_i = _consume_indented_continuation(lines, i + 1)
            desc_all = _join_fragments(([desc] if desc else []) + continuation)
            _ensure_field_section("params")
            bullet = f"- `{name}`"
            if param_type:
                bullet += f" (`{param_type}`)"
            if desc_all:
                bullet += f": {desc_all}"
            out.append(bullet)
            i = next_i
            continue

        if stripped.startswith(":raises "):
            exc, desc = _parse_sphinx_raises(stripped)
            continuation, next_i = _consume_indented_continuation(lines, i + 1)
            desc_all = _join_fragments(([desc] if desc else []) + continuation)
            _ensure_field_section("raises")
            bullet = f"- `{exc}`"
            if desc_all:
                bullet += f": {desc_all}"
            out.append(bullet)
            i = next_i
            continue

        if stripped.startswith(":return"):
            return_type, desc = _parse_sphinx_return(stripped)
            continuation, next_i = _consume_indented_continuation(lines, i + 1)
            desc_all = _join_fragments(([desc] if desc else []) + continuation)
            _ensure_field_section("returns")
            bullet = "- "
            if return_type:
                bullet += f"`{return_type}`"
                if desc_all:
                    bullet += f": {desc_all}"
            else:
                bullet += desc_all or "Return value."
            out.append(bullet)
            i = next_i
            continue

        if stripped.startswith(":rtype:"):
            rtype = stripped.split(":", maxsplit=2)[-1].strip()
            _ensure_field_section("returns")
            out.append(f"- Type: `{rtype}`")
            i += 1
            continue

        if current_field_section is not None and stripped:
            out.append("")
            current_field_section = None

        out.append(stripped)
        i += 1

    # Trim duplicate trailing blank lines while preserving paragraph separation.
    while len(out) >= 2 and out[-1] == "" and out[-2] == "":
        out.pop()
    return "\n".join(out).strip()


def _iter_public_names(module: ModuleType) -> list[str]:
    explicit = getattr(module, "__all__", None)
    if explicit is not None:
        return [str(name) for name in explicit]
    return [name for name in sorted(dir(module)) if not name.startswith("_")]


def _iter_class_members(cls: type[Any]) -> list[tuple[str, Any]]:
    members: list[tuple[str, Any]] = []
    for name, raw_member in cls.__dict__.items():
        if name.startswith("_"):
            continue

        member = raw_member
        if isinstance(raw_member, (staticmethod, classmethod)):
            member = raw_member.__func__

        if inspect.isfunction(member) or inspect.isbuiltin(member) or isinstance(raw_member, property):
            members.append((name, raw_member))

    return members


def _render_class(name: str, cls: type[Any]) -> list[str]:
    lines = [f"## `{name}`", ""]
    signature = _format_signature(cls, prefix=f"class {name}")
    if signature:
        lines.extend(_render_signature_block(signature))

    lines.extend([_format_docstring(cls), ""])

    members = _iter_class_members(cls)
    if members:
        lines.extend(["### Members", ""])

    for member_name, raw_member in members:
        if isinstance(raw_member, property):
            lines.extend([f"#### `{member_name}`", "", "`property`", "", _format_docstring(raw_member), ""])
            continue

        target = raw_member
        if isinstance(raw_member, (staticmethod, classmethod)):
            target = raw_member.__func__

        kind = "method"
        if isinstance(raw_member, staticmethod):
            kind = "staticmethod"
        elif isinstance(raw_member, classmethod):
            kind = "classmethod"

        signature = _format_signature(target, prefix=f"{member_name}")
        lines.append(f"#### `{member_name}`")
        lines.append("")
        if signature:
            lines.extend(_render_signature_block(signature))
        lines.append(f"`{kind}`")
        lines.append("")
        lines.append(_format_docstring(target))
        lines.append("")

    return lines


def _render_function(name: str, fn: Any) -> list[str]:
    lines = [f"## `{name}`", ""]
    signature = _format_signature(fn, prefix=name)
    if signature:
        lines.extend(_render_signature_block(signature))
    lines.extend([_format_docstring(fn), ""])
    return lines


def _render_value(name: str, value: Any) -> list[str]:
    lines = [f"## `{name}`", "", "`value`", "", _format_docstring(value), ""]
    return lines


def _render_module_page(module_name: str) -> str:
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"Could not import '{module_name}'. Install the package into the active environment "
            "first (for example: `pip install -e .`)."
        ) from exc
    title = module_name.split(".")[-1].replace("_", " ").title()

    lines: list[str] = [
        f"# {title} API",
        "",
        "_This file is auto-generated by `tools/generate_api_docs.py`. Do not edit manually._",
        "",
        f"## Module `{module_name}`",
        "",
        _format_docstring(module),
        "",
    ]

    for name in _iter_public_names(module):
        if not hasattr(module, name):
            continue
        value = getattr(module, name)

        if inspect.isclass(value):
            lines.extend(_render_class(name, value))
        elif inspect.isfunction(value) or inspect.isbuiltin(value):
            lines.extend(_render_function(name, value))
        else:
            lines.extend(_render_value(name, value))

    return "\n".join(lines).rstrip() + "\n"


def _parse_module_args(values: list[str] | None) -> dict[str, str]:
    if not values:
        return DEFAULT_MODULE_MAP

    parsed: dict[str, str] = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"Invalid --module '{item}'. Expected '<slug>=<module.path>'.")
        slug, module_name = item.split("=", maxsplit=1)
        slug = slug.strip()
        module_name = module_name.strip()
        if not slug or not module_name:
            raise ValueError(f"Invalid --module '{item}'. Expected '<slug>=<module.path>'.")
        parsed[slug] = module_name

    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("docs/api"),
        help="Output directory for generated Markdown API pages.",
    )
    parser.add_argument(
        "--module",
        action="append",
        default=None,
        help="Module mapping in '<slug>=<module.path>' form. Can be passed multiple times.",
    )
    args = parser.parse_args()

    module_map = _parse_module_args(args.module)

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    for slug, module_name in module_map.items():
        output_path = out_dir / f"{slug}.md"
        output_path.write_text(_render_module_page(module_name), encoding="utf-8")
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
