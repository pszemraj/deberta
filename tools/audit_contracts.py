#!/usr/bin/env python3
"""
tools/audit_contracts.py

Contract-level audit harness for encoder modeling + pretraining pipeline.

This is objective-agnostic: it checks *universal* invariants that silently corrupt learning:
- forward topology is faithful (norm placement, FFN gating, residual scaling order)
- attention masks do what they claim under variable-length and packed/doc-block patterns
- loss token sets, labels, reductions, and gradient boundaries match the objective contract
- export/checkpoint utilities do not silently drop weights

Run (from repo root):
  python tools/audit_contracts.py
CI mode (fail on WARN/SKIP):
  python tools/audit_contracts.py --strict

This script is intentionally small-batch / CPU-friendly. It does not run training.
"""

from __future__ import annotations

import argparse
import ast
import random
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

# -----------------------------
# Reporting primitives
# -----------------------------

_STATUS_ORDER = {"PASS": 0, "WARN": 1, "SKIP": 2, "FAIL": 3}


@dataclass
class CheckResult:
    name: str
    status: str  # PASS | WARN | SKIP | FAIL
    details: str = ""
    hint: str = ""

    def sort_key(self) -> tuple[int, str]:
        return (_STATUS_ORDER.get(self.status, 99), self.name)


def _fmt_block(s: str, indent: str = "  ") -> str:
    s = (s or "").rstrip()
    if not s:
        return ""
    return "\n".join(indent + line for line in s.splitlines())


def _print_result(r: CheckResult, *, verbose: bool) -> None:
    head = f"[{r.status}] {r.name}"
    print(head)
    if verbose and r.details:
        print(_fmt_block(r.details))
    if r.status in {"FAIL", "WARN"} and r.hint:
        print(_fmt_block("Fix: " + r.hint))


def _fail(name: str, details: str, hint: str = "") -> CheckResult:
    return CheckResult(name=name, status="FAIL", details=details, hint=hint)


def _warn(name: str, details: str, hint: str = "") -> CheckResult:
    return CheckResult(name=name, status="WARN", details=details, hint=hint)


def _pass(name: str, details: str = "") -> CheckResult:
    return CheckResult(name=name, status="PASS", details=details)


def _skip(name: str, details: str = "") -> CheckResult:
    return CheckResult(name=name, status="SKIP", details=details)


# -----------------------------
# Repo helpers
# -----------------------------


def _repo_root_from_script() -> Path:
    # tools/audit_contracts.py -> tools -> repo root
    return Path(__file__).resolve().parent.parent


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _parse_ast(path: Path) -> ast.AST:
    return ast.parse(_read_text(path), filename=str(path))


def _extract_function_source(path: Path, func_name: str) -> str:
    src = _read_text(path)
    tree = ast.parse(src, filename=str(path))
    lines = src.splitlines()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            if not hasattr(node, "lineno") or not hasattr(node, "end_lineno"):
                raise RuntimeError("Python AST nodes missing lineno/end_lineno; need Python 3.8+.")
            start = int(node.lineno) - 1
            end = int(node.end_lineno)
            return "\n".join(lines[start:end])
    raise KeyError(f"Function {func_name!r} not found in {path}")


# -----------------------------
# Minimal tokenizer stub (no internet / no HF download required)
# -----------------------------


def TinyTokenizer(vocab_size: int = 128) -> Any:
    """Build shared tokenizer stub for audit checks.

    :param int vocab_size: Vocabulary size.
    :return Any: Tokenizer-like stub.
    """
    repo_root = _repo_root_from_script()
    tests_path = str(repo_root / "tests")
    if tests_path not in sys.path:
        sys.path.insert(0, tests_path)
    from _fakes import DummyTokenizer

    token_map: dict[int, str] = {}
    for idx in range(5, int(vocab_size)):
        if idx % 11 == 0:
            token_map[idx] = "##ing"
        elif idx % 7 == 0:
            token_map[idx] = "##ly"
        else:
            token_map[idx] = f"t{idx}"
    tok = DummyTokenizer(
        vocab_size=vocab_size,
        token_map=token_map,
        default_token_prefix="##",
        vocab_type="wordpiece",
    )
    tok.unk_token = "[UNK]"
    tok.unk_token_id = 4
    tok._id_to_tok[tok.unk_token_id] = tok.unk_token
    tok.all_special_ids = [
        tok.pad_token_id,
        tok.cls_token_id,
        tok.sep_token_id,
        tok.mask_token_id,
        tok.unk_token_id,
    ]
    tok.all_special_tokens = ["[PAD]", "[CLS]", "[SEP]", "[MASK]", "[UNK]"]

    def _tokenize_wordpiece_probe(text: str) -> list[str]:
        out: list[str] = []
        for word in str(text).strip().split():
            out.append(word)
            out.append("##x")
        return out

    tok.tokenize = _tokenize_wordpiece_probe  # type: ignore[assignment]
    return tok


# -----------------------------
# Checks
# -----------------------------


def check_repo_layout(repo_root: Path) -> CheckResult:
    required = [
        repo_root / "src" / "deberta" / "data" / "collator.py",
        repo_root / "src" / "deberta" / "modeling" / "rtd.py",
        repo_root / "src" / "deberta" / "training" / "entrypoint.py",
        repo_root / "src" / "deberta" / "training" / "compile.py",
        repo_root / "src" / "deberta" / "export_cli.py",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        return _fail(
            "repo_layout",
            "Missing expected files:\n" + "\n".join(missing),
            hint="Run this script from the DeBERTa repo root (the directory that contains src/ and tools/).",
        )
    return _pass("repo_layout")


def check_export_cli_ignores_missing_keys(repo_root: Path) -> CheckResult:
    """
    export_cli.py currently calls load_intersection_state_dict(...) and ignores the returned
    IncompatibleKeys, which is the definition of a silent partial-load failure.
    """
    path = repo_root / "src" / "deberta" / "export_cli.py"
    tree = _parse_ast(path)

    ignored_calls: list[tuple[int, str]] = []

    class Visitor(ast.NodeVisitor):
        def visit_Expr(self, node: ast.Expr) -> Any:
            if isinstance(node.value, ast.Call):
                fn = node.value.func
                if isinstance(fn, ast.Name) and fn.id == "load_intersection_state_dict":
                    ignored_calls.append((getattr(node, "lineno", -1), "load_intersection_state_dict(...)"))
                elif isinstance(fn, ast.Attribute) and fn.attr == "load_intersection_state_dict":
                    ignored_calls.append((getattr(node, "lineno", -1), "X.load_intersection_state_dict(...)"))
            self.generic_visit(node)

    Visitor().visit(tree)

    if ignored_calls:
        lines = "\n".join(f"  - line {ln}: {expr}" for ln, expr in ignored_calls)
        return _fail(
            "export_cli_strict_loading",
            "export_cli.py ignores the return value of load_intersection_state_dict(). "
            "This can silently drop weights on export.\n"
            f"{lines}",
            hint=(
                "Capture the IncompatibleKeys return, and fail fast on missing_keys unless they are explicitly "
                "allowlisted for that export mode (tied embeddings, intentionally omitted heads, etc.)."
            ),
        )
    return _pass("export_cli_strict_loading")


def check_collator_whole_word_fallback(repo_root: Path) -> CheckResult:
    """
    Whole-word n-gram masking contract: if you claim 'whole-word', you must never mask only
    a subset of a multi-subtoken word.

    This collator has an explicit token-level fallback when the n-gram sampler can't fill
    the masking budget. That is not always wrong, but it is a silent spec regression if you
    *claim* strict whole-word masking.
    """
    path = repo_root / "src" / "deberta" / "data" / "collator.py"
    src = _read_text(path)

    # Heuristic signature of the fallback:
    sig = "remaining_candidates = [idx for idx in maskable_set if idx not in masked_set]"
    if sig in src:
        return _warn(
            "collator_whole_word_strictness",
            "DebertaV3ElectraCollator._mask_tokens_ngram() falls back to masking individual token indices when "
            "it cannot fill the n-gram budget. This can mask only part of a multi-subtoken word.",
            hint=(
                "If you require strict whole-word masking, change the fallback to operate on whole word-groups "
                "(or reduce num_to_mask until it is feasible). If you do *not* require strict WWM, document this "
                "behavior explicitly because it changes the learning signal."
            ),
        )
    return _pass("collator_whole_word_strictness")


def check_collator_determinism(repo_root: Path) -> CheckResult:
    """
    Determinism contract: given identical RNG seeds and identical input features, the collator must produce
    bit-identical masked inputs + labels.

    This is critical for debugging and for objective correctness tests that rely on reproducible corruption.
    """
    try:
        from deberta.data.collator import DebertaV3ElectraCollator, MLMConfig
    except Exception as e:
        return _fail(
            "collator_determinism",
            f"Failed to import collator: {type(e).__name__}: {e}",
            hint="Ensure repo src/ is importable.",
        )

    tokenizer = TinyTokenizer(vocab_size=128)
    cfg = MLMConfig(mlm_probability=0.35, mask_token_prob=0.8, random_token_prob=0.1, max_ngram=1)
    collator = DebertaV3ElectraCollator(
        tokenizer=tokenizer,
        cfg=cfg,
        packed_sequences=False,
        block_cross_document_attention=False,
        pad_to_multiple_of=None,
    )

    ex1 = [tokenizer.cls_token_id, 5, 6, 7, 8, tokenizer.sep_token_id]
    ex2 = [tokenizer.cls_token_id, 9, 10, tokenizer.sep_token_id]
    features = [{"input_ids": ex1}, {"input_ids": ex2}]

    _seed_everything(42)
    out1 = collator(features)
    _seed_everything(42)
    out2 = collator(features)

    # Compare a stable subset of keys.
    keys = sorted(set(out1.keys()) & set(out2.keys()))
    mismatches = []
    for k in keys:
        v1, v2 = out1[k], out2[k]
        if isinstance(v1, torch.Tensor) and isinstance(v2, torch.Tensor):
            if not torch.equal(v1, v2):
                mismatches.append(k)
        else:
            if v1 != v2:
                mismatches.append(k)

    if mismatches:
        return _fail(
            "collator_determinism",
            "Collator outputs differ across two calls with identical seeds. Mismatched keys: "
            + ", ".join(mismatches),
            hint="Ensure corruption uses torch RNG only (not python.random / NumPy), and that seed-setting is applied consistently.",
        )

    return _pass("collator_determinism")


def check_optimizer_state_ordering_risk(repo_root: Path) -> CheckResult:
    """
    PyTorch AdamW state_dict maps momentum/variance to parameters by *group order*, not name.
    If parameter registration order changes between code versions, resume can silently corrupt
    optimizer state.

    If the codebase has a param-order digest mechanism (persist + validate on resume), this is
    mitigated; otherwise it is a silent risk.
    """
    runtime_path = repo_root / "src" / "deberta" / "training" / "runtime.py"
    entrypoint_path = repo_root / "src" / "deberta" / "training" / "entrypoint.py"
    src = _read_text(runtime_path) + "\n" + _read_text(entrypoint_path)

    # Check if param-order digest mechanism is present.
    has_digest = "_optimizer_param_order_digest" in src and "optimizer_param_digest" in src
    if has_digest:
        return _pass(
            "optimizer_resume_param_order_risk",
            details="Optimizer param-order digest is persisted in checkpoints and validated on resume.",
        )

    # Heuristic: optimizer builder iterates model.named_parameters() without sorting or explicit name->state mapping.
    if "for name, param in model.named_parameters()" in src and "sorted(" not in src:
        return _warn(
            "optimizer_resume_param_order_risk",
            "Optimizer param groups are built from model.named_parameters() order (no name-sorted canonicalization). "
            "This is the standard PyTorch behavior, but it means 'resume is identical' is not robust to refactors "
            "that change parameter registration order.",
            hint=(
                "Persist and validate a digest of parameter *names in optimizer order* inside each checkpoint. "
                "On resume, recompute and compare; if mismatch, fail fast (or implement a name-based optimizer-state "
                "remap utility)."
            ),
        )
    return _pass("optimizer_resume_param_order_risk")


def _exec_extracted_function(path: Path, func_name: str, globals_dict: dict[str, Any]) -> Callable[..., Any]:
    src = _extract_function_source(path, func_name)
    # Compile in an isolated module-like dict.
    loc: dict[str, Any] = {}
    g = dict(globals_dict)
    exec(compile(src, filename=str(path), mode="exec"), g, loc)
    fn = loc.get(func_name) or g.get(func_name)
    if not callable(fn):
        raise RuntimeError(f"Failed to exec function {func_name} from {path}")
    return fn  # type: ignore[return-value]


def check_doc_block_mask_contract(repo_root: Path) -> CheckResult:
    path = repo_root / "src" / "deberta" / "training" / "compile.py"
    g = {
        "torch": torch,
        "_DOC_BLOCK_EYE_CACHE": {},
        "_DOC_BLOCK_CLS_KEY_CACHE": {},
    }
    try:
        build_mask = _exec_extracted_function(path, "_build_doc_block_mask", g)
    except Exception as e:
        return _fail(
            "doc_block_mask_contract",
            f"Failed to load _build_doc_block_mask via AST exec: {type(e).__name__}: {e}",
            hint="This harness expects src/deberta/training/compile.py to define _build_doc_block_mask(doc_ids).",
        )

    # Build a small doc-id batch.
    doc_ids = torch.tensor(
        [
            [1, 1, 1, 1, 2, 2, 0, 0],
            [1, 1, 2, 2, 2, 0, 0, 0],
        ],
        dtype=torch.long,
    )
    keep = build_mask(doc_ids)

    if keep.dtype != torch.bool:
        return _fail(
            "doc_block_mask_contract",
            f"_build_doc_block_mask must return torch.bool, got dtype={keep.dtype}",
            hint="Return a boolean keep-mask (True=keep edge).",
        )
    if keep.shape != (2, 8, 8):
        return _fail(
            "doc_block_mask_contract",
            f"Unexpected keep-mask shape: expected (2,8,8), got {tuple(keep.shape)}",
        )

    active = doc_ids.ne(0)
    diag = torch.diagonal(keep, dim1=1, dim2=2)
    if not torch.equal(diag, active):
        return _fail(
            "doc_block_mask_contract",
            "Diagonal of keep-mask must equal active queries (doc_id != 0).",
            hint="Ensure keep[i,i]=True for active tokens and False for inactive/pad tokens.",
        )

    # Active tokens must not attend across docs.
    for b in range(doc_ids.shape[0]):
        for i in range(doc_ids.shape[1]):
            for j in range(doc_ids.shape[1]):
                if i == j:
                    continue
                ai = bool(active[b, i].item())
                aj = bool(active[b, j].item())
                if ai and aj:
                    same = int(doc_ids[b, i].item()) == int(doc_ids[b, j].item())
                    if bool(keep[b, i, j].item()) != same:
                        return _fail(
                            "doc_block_mask_contract",
                            f"Cross-doc leakage at batch={b}, i={i}, j={j}: doc_ids=({int(doc_ids[b, i])},{int(doc_ids[b, j])}) keep={bool(keep[b, i, j])}",
                            hint="keep should be True iff both tokens are active and in the same document.",
                        )

    # Inactive rows must keep exactly CLS key 0 and nothing else.
    for b in range(doc_ids.shape[0]):
        for i in range(doc_ids.shape[1]):
            if not bool(active[b, i].item()):
                row = keep[b, i]
                if not bool(row[0].item()):
                    return _fail(
                        "doc_block_mask_contract",
                        f"Inactive query row batch={b}, i={i} must keep CLS key 0 (to avoid all-False SDPA rows).",
                    )
                if bool(row[1:].any().item()):
                    return _fail(
                        "doc_block_mask_contract",
                        f"Inactive query row batch={b}, i={i} must not keep any non-CLS keys.",
                        hint="keep should contain a single True at key position 0 for inactive queries.",
                    )

    return _pass("doc_block_mask_contract")


def check_attention_mask_to_active_tokens_contract(repo_root: Path) -> CheckResult:
    path = repo_root / "src" / "deberta" / "modeling" / "rtd.py"
    try:
        from deberta.modeling.mask_utils import normalize_keep_mask
    except ImportError:
        normalize_keep_mask = None  # type: ignore[assignment]
    g: dict[str, Any] = {"torch": torch}
    if normalize_keep_mask is not None:
        g["normalize_keep_mask"] = normalize_keep_mask
    try:
        fn = _exec_extracted_function(path, "attention_mask_to_active_tokens", g)
    except Exception as e:
        return _fail(
            "attention_mask_to_active_tokens_contract",
            f"Failed to load attention_mask_to_active_tokens via AST exec: {type(e).__name__}: {e}",
        )

    B, S = 2, 5
    pad = 0
    input_ids = torch.tensor([[1, 9, 9, 0, 0], [1, 9, 0, 0, 0]], dtype=torch.long)

    # None mask: should use pad_token_id.
    active = fn(input_ids=input_ids, attention_mask=None, pad_token_id=pad)
    if active.shape != (B, S) or active.dtype != torch.bool:
        return _fail(
            "attention_mask_to_active_tokens_contract", "Unexpected output shape/dtype for None mask."
        )
    if not torch.equal(active, input_ids.ne(pad)):
        return _fail(
            "attention_mask_to_active_tokens_contract",
            "None mask path must return input_ids != pad_token_id when pad_token_id is provided.",
        )

    # 2D mask.
    mask2 = torch.tensor([[1, 1, 0, 0, 0], [1, 0, 0, 0, 0]], dtype=torch.long)
    active2 = fn(input_ids=input_ids, attention_mask=mask2, pad_token_id=pad)
    if not torch.equal(active2, mask2.to(torch.bool)):
        return _fail(
            "attention_mask_to_active_tokens_contract", "2D attention mask must be returned as bool."
        )

    # 3D mask: diagonal activity.
    mask3 = torch.zeros((B, S, S), dtype=torch.bool)
    mask3[:, 0, 0] = True
    mask3[:, 1, 1] = True
    active3 = fn(input_ids=input_ids, attention_mask=mask3, pad_token_id=pad)
    expected3 = torch.tensor([[1, 1, 0, 0, 0], [1, 1, 0, 0, 0]], dtype=torch.bool) & input_ids.ne(pad)
    if not torch.equal(active3, expected3):
        return _fail(
            "attention_mask_to_active_tokens_contract",
            "3D mask path must use diagonal as query activity, AND must AND with pad mask when pad_token_id is provided.",
        )

    # 4D broadcast SDPA-style: (B,1,1,S)
    mask4 = mask2.to(torch.bool)[:, None, None, :]
    active4 = fn(input_ids=input_ids, attention_mask=mask4, pad_token_id=pad)
    if not torch.equal(active4, mask2.to(torch.bool) & input_ids.ne(pad)):
        return _fail(
            "attention_mask_to_active_tokens_contract",
            "4D broadcast path must return per-token activity mask (B,S).",
        )

    # Float/additive masks must be *rejected* (not silently cast to bool).
    # normalize_keep_mask raises ValueError for floating-point masks.
    add = torch.zeros((B, S), dtype=torch.float32)
    add[:, 2:] = -1e4  # masked positions in additive convention
    try:
        _ = fn(input_ids=input_ids, attention_mask=add, pad_token_id=pad)
        return _warn(
            "attention_mask_to_active_tokens_contract",
            "attention_mask_to_active_tokens silently accepts float masks. "
            "Additive float masks (0/-1e4) become all-True under bool cast, corrupting attention.",
            hint="Reject floating-point masks at the boundary to prevent silent mask corruption.",
        )
    except (ValueError, TypeError):
        pass  # Expected: normalize_keep_mask rejects float masks.

    return _pass("attention_mask_to_active_tokens_contract")


def _try_import_transformers() -> tuple[bool, str]:
    try:
        import transformers  # noqa: F401

        return True, ""
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def check_rope_layer_norm_order(repo_root: Path) -> CheckResult:
    ok, err = _try_import_transformers()
    if not ok:
        return _skip(
            "rope_layer_norm_order",
            f"transformers not available ({err}); skipping model-level forward checks.",
        )

    try:
        from deberta.modeling.rope_encoder import DebertaRoPEConfig, DebertaRoPELayer
    except Exception as e:
        return _fail(
            "rope_layer_norm_order",
            f"Failed to import DebertaRoPELayer: {type(e).__name__}: {e}",
            hint="Ensure transformers is installed and src/ is on PYTHONPATH.",
        )

    def _run(norm_arch: str) -> tuple[bool, str]:
        cfg = DebertaRoPEConfig(
            vocab_size=128,
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            intermediate_size=64,
            hidden_act="gelu",
            ffn_type="swiglu",
            use_bias=False,
            max_position_embeddings=64,
            type_vocab_size=2,
            norm_arch=norm_arch,
            attention_implementation="sdpa",
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
            norm_eps=1e-6,
            keel_alpha_init=4.0,
            keel_alpha_learnable=True,
        )
        layer = DebertaRoPELayer(cfg, alpha_init=4.0)
        layer.eval()

        events: list[str] = []

        def hook(name: str) -> Callable[..., Any]:
            def _h(_m: torch.nn.Module, _inp: tuple[Any, ...], _out: Any) -> None:
                events.append(name)

            return _h

        # Attach hooks based on arch.
        handles = []
        if norm_arch == "post":
            handles.append(layer.attn.register_forward_hook(hook("attn")))
            handles.append(layer.dropout.register_forward_hook(hook("dropout")))
            handles.append(layer.norm1.register_forward_hook(hook("norm1")))
            handles.append(layer.mlp.register_forward_hook(hook("mlp")))
            handles.append(layer.norm2.register_forward_hook(hook("norm2")))
        else:
            handles.append(layer.inner_norm1.register_forward_hook(hook("inner_norm1")))
            handles.append(layer.attn.register_forward_hook(hook("attn")))
            handles.append(layer.alpha1.register_forward_hook(hook("alpha1")))
            handles.append(layer.dropout.register_forward_hook(hook("dropout")))
            handles.append(layer.outer_norm1.register_forward_hook(hook("outer_norm1")))
            handles.append(layer.inner_norm2.register_forward_hook(hook("inner_norm2")))
            handles.append(layer.mlp.register_forward_hook(hook("mlp")))
            handles.append(layer.alpha2.register_forward_hook(hook("alpha2")))
            handles.append(layer.outer_norm2.register_forward_hook(hook("outer_norm2")))

        try:
            _seed_everything(0)
            x = torch.randn(2, 8, 32)
            mask = torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 0, 0, 0]], dtype=torch.long)
            _ = layer(x, mask)
        finally:
            for h in handles:
                h.remove()

        if norm_arch == "post":
            expected = ["attn", "dropout", "norm1", "mlp", "dropout", "norm2"]
            # Dropout hook is single module used twice; our hook name appears twice.
            if events != expected:
                return False, f"Expected call order {expected}, got {events}"
        else:
            expected_prefix = ["inner_norm1", "attn", "alpha1", "dropout", "outer_norm1"]
            expected_suffix = ["inner_norm2", "mlp", "alpha2", "dropout", "outer_norm2"]
            if events[:5] != expected_prefix or events[5:] != expected_suffix:
                return False, f"Expected call order {expected_prefix + expected_suffix}, got {events}"
        return True, "ok"

    ok_post, msg_post = _run("post")
    ok_keel, msg_keel = _run("keel")

    if not ok_post or not ok_keel:
        return _fail(
            "rope_layer_norm_order",
            f"Norm placement / call order mismatch.\npost: {msg_post}\nkeel: {msg_keel}",
            hint="A single misplaced RMSNorm is a silent performance regression. Fix the forward composition in DebertaRoPELayer.",
        )
    return _pass("rope_layer_norm_order")


def check_rope_attention_mask_leak(repo_root: Path) -> CheckResult:
    ok, err = _try_import_transformers()
    if not ok:
        return _skip(
            "rope_attention_mask_leak", f"transformers not available ({err}); skipping attention leak checks."
        )

    try:
        from deberta.modeling.rope_encoder import DebertaRoPEConfig, DebertaRoPESelfAttention
    except Exception as e:
        return _fail(
            "rope_attention_mask_leak",
            f"Failed to import DebertaRoPESelfAttention: {type(e).__name__}: {e}",
        )

    # Prefer testing the *repo's* packed/doc-block mask builder, without importing the whole training module.
    build_mask = None
    try:
        build_mask = _exec_extracted_function(
            repo_root / "src" / "deberta" / "training" / "compile.py",
            "_build_doc_block_mask",
            {"torch": torch, "_DOC_BLOCK_EYE_CACHE": {}, "_DOC_BLOCK_CLS_KEY_CACHE": {}},
        )
    except Exception:
        build_mask = None  # We'll fall back to a minimal within-doc mask (no padding edge cases).

    cfg = DebertaRoPEConfig(
        vocab_size=128,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=64,
        hidden_act="gelu",
        ffn_type="swiglu",
        use_bias=False,
        max_position_embeddings=64,
        type_vocab_size=2,
        norm_arch="post",
        attention_implementation="sdpa",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        norm_eps=1e-6,
    )
    attn = DebertaRoPESelfAttention(cfg).eval()

    _seed_everything(0)
    x = torch.randn(2, 8, 32, requires_grad=True)

    # 2D padding mask: last 3 tokens in batch0 are pad (masked off).
    mask2 = torch.tensor(
        [
            [1, 1, 1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 0, 0],
        ],
        dtype=torch.long,
    )

    y = attn(x, mask2)
    # Padded queries must be exactly zeroed out.
    if float(y[0, 6].abs().max().item()) > 1e-7:
        return _fail(
            "rope_attention_mask_leak",
            "2D mask: padded query positions should be zeroed (query_keep).",
            hint="Verify query_keep is applied after output projection, and that 2D masks are treated as key-keep masks.",
        )

    loss = y[0, 0].sum()
    loss.backward(retain_graph=True)
    grad0 = x.grad.detach().clone()
    x.grad.zero_()

    masked_positions = (mask2[0] == 0).nonzero(as_tuple=False).view(-1).tolist()
    for j in masked_positions:
        gnorm = float(grad0[0, j].abs().max().item())
        if gnorm > 1e-7:
            return _fail(
                "rope_attention_mask_leak",
                f"2D mask leak: output at (batch0,pos0) depends on masked key pos{j} (max|grad|={gnorm:.2e}).",
                hint="Mask broadcasting or dtype may be wrong. Verify SDPA/eager mask contract (bool keep-mask vs additive).",
            )

    # 3D doc-block mask: test cross-document leakage + padded edge cases.
    if build_mask is not None:
        doc_ids = torch.tensor([[1, 1, 1, 2, 2, 0, 0, 0]], dtype=torch.long)
        keep = build_mask(doc_ids)
    else:
        # Fallback: simple within-doc mask (no special CLS edges for inactive queries).
        doc_ids = torch.tensor([[1, 1, 1, 1, 2, 2, 2, 2]], dtype=torch.long)
        keep = (
            doc_ids[:, :, None].eq(doc_ids[:, None, :])
            & doc_ids.ne(0)[:, :, None]
            & doc_ids.ne(0)[:, None, :]
        )
        eye = torch.eye(8, dtype=torch.bool)
        keep = (keep & ~eye[None, :, :]) | (eye[None, :, :] & doc_ids.ne(0)[:, :, None])

    x2 = torch.randn(1, 8, 32, requires_grad=True)
    y2 = attn(x2, keep)

    # Padded queries (inactive diag) must be exactly zeroed out (including out_proj bias).
    if float(y2[0, 6].abs().max().item()) > 1e-7:
        return _fail(
            "rope_attention_mask_leak",
            "3D mask: inactive query positions should be zeroed (query_keep).",
            hint="For 3D masks, diagonal must encode query activity; ensure query_keep_tokens multiplies output.",
        )

    loss2 = y2[0, 1].sum()  # query in doc1
    loss2.backward()
    grad = x2.grad.detach()

    # Keys in other docs and padding must not influence.
    blocked = [3, 4, 5, 6, 7] if build_mask is not None else list(range(4, 8))
    for j in blocked:
        gnorm = float(grad[0, j].abs().max().item())
        if gnorm > 1e-7:
            return _fail(
                "rope_attention_mask_leak",
                f"3D mask leak: output at (pos1/doc1) depends on blocked key pos{j} (max|grad|={gnorm:.2e}).",
                hint="Doc-block keep-mask must be (B,S,S) bool; cross-doc and pad keys must be masked for active queries.",
            )

    return _pass("rope_attention_mask_leak")


def check_rope_ffn_gating(repo_root: Path) -> CheckResult:
    path = repo_root / "src" / "deberta" / "modeling" / "rope_encoder.py"
    src = _read_text(path)

    # Fast static sanity: SwiGLU path must chunk the projection into 2 and multiply silu(gate)*up.
    has_chunk = ".chunk(2" in src and "F.silu" in src and "* up" in src
    if not has_chunk:
        return _fail(
            "rope_ffn_gating",
            "Could not find a fused split-and-gate SwiGLU pattern in DebertaRoPEMLP.forward().",
            hint="SwiGLU requires projecting to 2*intermediate, splitting into (gate, up), applying silu to gate, then gate*up.",
        )
    return _pass("rope_ffn_gating")


def check_rtd_loss_integrity(repo_root: Path, *, verbose: bool) -> CheckResult:
    ok, err = _try_import_transformers()
    if not ok:
        return _skip(
            "rtd_loss_integrity", f"transformers not available ({err}); skipping RTD forward/loss checks."
        )

    try:
        from deberta.data.collator import DebertaV3ElectraCollator, MLMConfig
        from deberta.modeling import DebertaRoPEConfig, DebertaRoPEModel, DebertaV3RTDPretrainer
    except Exception as e:
        return _fail(
            "rtd_loss_integrity",
            f"Failed to import RTD/collator modules: {type(e).__name__}: {e}",
            hint="Ensure repo src/ is importable and required deps (transformers) are installed.",
        )

    # Use a local stub tokenizer to avoid network downloads.
    tokenizer = TinyTokenizer(vocab_size=128)

    mlm_cfg = MLMConfig(
        mlm_probability=0.35,
        mask_token_prob=0.8,
        random_token_prob=0.1,
        max_ngram=1,
    )
    collator = DebertaV3ElectraCollator(
        tokenizer=tokenizer,
        cfg=mlm_cfg,
        packed_sequences=False,
        block_cross_document_attention=False,
        pad_to_multiple_of=None,
    )

    # Two examples with padding and special tokens.
    # Format: [CLS] t5 t6 t7 t8 [SEP] [PAD] ...
    ex1 = [tokenizer.cls_token_id, 5, 6, 7, 8, tokenizer.sep_token_id]
    ex2 = [tokenizer.cls_token_id, 9, 10, tokenizer.sep_token_id]
    features = [{"input_ids": ex1}, {"input_ids": ex2}]
    orig_padded_ids = tokenizer.pad(features, return_tensors="pt")["input_ids"].clone()
    batch = collator(features)
    input_ids = batch["input_ids"]
    labels = batch["labels"]
    attention_mask = batch.get("attention_mask", None)

    # Basic label correctness: labels must come from the *original* (pre-corruption) token ids.
    if labels.shape != input_ids.shape:
        return _fail("rtd_loss_integrity", "Collator produced labels with wrong shape.")

    masked_pos = labels.ne(-100)
    if bool(masked_pos.any().item()):
        if not torch.equal(labels[masked_pos], orig_padded_ids[masked_pos]):
            return _fail(
                "rtd_loss_integrity",
                "MLM labels do not match original (pre-corruption) token ids at supervised positions.",
                hint="Labels must be captured *before* any masking/replacement happens.",
            )

    # Original special tokens (CLS/SEP/PAD/...) must never be supervised or corrupted.
    special_ids = set(int(x) for x in tokenizer.all_special_ids)
    for b in range(orig_padded_ids.shape[0]):
        for i in range(orig_padded_ids.shape[1]):
            orig_tid = int(orig_padded_ids[b, i].item())
            if orig_tid in special_ids:
                if int(labels[b, i].item()) != -100:
                    return _fail(
                        "rtd_loss_integrity",
                        f"Collator produced a supervised label on an original special token at (b={b},i={i},id={orig_tid}).",
                        hint="Special tokens must be excluded from MLM supervision and corruption.",
                    )
                if int(input_ids[b, i].item()) != orig_tid:
                    return _fail(
                        "rtd_loss_integrity",
                        f"Collator corrupted an original special token at (b={b},i={i},id={orig_tid}).",
                        hint="CLS/SEP/PAD must remain intact (never replaced by [MASK]/random tokens).",
                    )

    # Build tiny RoPE backbones for gen/disc.

    # Ensure special token ids are set on config so RTD forbidden sampling mask is correct.
    def _mk_cfg() -> DebertaRoPEConfig:
        return DebertaRoPEConfig(
            vocab_size=tokenizer.vocab_size,
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=64,
            hidden_act="gelu",
            ffn_type="swiglu",
            use_bias=False,
            max_position_embeddings=64,
            type_vocab_size=2,
            norm_arch="post",
            attention_implementation="sdpa",
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
            norm_eps=1e-6,
            pad_token_id=tokenizer.pad_token_id,
            cls_token_id=tokenizer.cls_token_id,
            sep_token_id=tokenizer.sep_token_id,
            mask_token_id=tokenizer.mask_token_id,
            unk_token_id=tokenizer.unk_token_id,
        )

    gen_cfg = _mk_cfg()
    disc_cfg = _mk_cfg()

    generator = DebertaRoPEModel(gen_cfg)
    discriminator = DebertaRoPEModel(disc_cfg)

    model = DebertaV3RTDPretrainer(
        discriminator_backbone=discriminator,
        generator_backbone=generator,
        disc_config=disc_cfg,
        gen_config=gen_cfg,
        embedding_sharing="none",
        tie_generator_word_embeddings=True,
    )
    model.train()

    # Capture gen_logits, disc_logits, and sampled ids by patching/hooking.
    captured: dict[str, Any] = {"gen_logits": None, "disc_logits": None, "sampled": None}

    # Patch gumbel sampler to capture sampled ids.
    orig_gumbel = model._gumbel_sample

    def _gumbel_wrapper(
        logits: torch.Tensor, *, temperature: float, forbidden_vocab_mask: torch.Tensor | None
    ) -> torch.Tensor:
        out = orig_gumbel(logits, temperature=temperature, forbidden_vocab_mask=forbidden_vocab_mask)
        captured["sampled"] = out.detach().clone()
        return out

    model._gumbel_sample = _gumbel_wrapper  # type: ignore[assignment]

    h1 = model.generator_lm_head.register_forward_hook(
        lambda _m, _inp, out: captured.__setitem__("gen_logits", out.detach().clone())
    )
    h2 = model.discriminator_head.register_forward_hook(
        lambda _m, _inp, out: captured.__setitem__("disc_logits", out.detach().clone())
    )

    try:
        _seed_everything(123)
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            token_type_ids=batch.get("token_type_ids", None),
            sampling_temperature=1.0,
            gen_loss_weight=1.0,
            disc_loss_weight=50.0,
        )
    finally:
        h1.remove()
        h2.remove()
        model._gumbel_sample = orig_gumbel  # type: ignore[assignment]

    if captured["gen_logits"] is None or captured["disc_logits"] is None or captured["sampled"] is None:
        return _fail(
            "rtd_loss_integrity",
            "Failed to capture gen_logits/disc_logits/sampled ids; hooks/patching did not fire.",
            hint="This indicates an unexpected control-flow path in RTD forward. Check masked_positions logic and hooks.",
        )

    gen_logits = captured["gen_logits"]
    disc_logits = captured["disc_logits"]
    sampled = captured["sampled"]

    # Recompute masked positions and loss terms exactly as implementation.
    masked_positions = labels.ne(-100)
    masked_flat = masked_positions.view(-1)
    masked_idx = torch.nonzero(masked_flat, as_tuple=False).squeeze(-1)
    labels_flat = labels.view(-1)
    masked_labels = labels_flat.index_select(0, masked_idx)

    # Generator loss: CE over masked positions only.
    if gen_logits.shape[0] != masked_idx.numel():
        return _fail(
            "rtd_loss_integrity",
            f"Generator logits shape mismatch: expected first dim {masked_idx.numel()}, got {gen_logits.shape[0]}",
            hint="Generator MLM head must be computed only on supervised token positions.",
        )

    gen_loss_ref = torch.nn.functional.cross_entropy(gen_logits.float(), masked_labels)
    if not torch.allclose(gen_loss_ref, out.gen_loss_raw, atol=1e-6, rtol=0):
        return _fail(
            "rtd_loss_integrity",
            f"Generator loss mismatch: ref={float(gen_loss_ref):.6f} impl={float(out.gen_loss_raw):.6f}",
            hint="Check supervised token-set selection and CE reduction.",
        )

    # Discriminator labels: replaced iff sampled != original label at masked positions.
    replaced = sampled.to(dtype=masked_labels.dtype).ne(masked_labels).to(dtype=torch.float32)
    disc_labels_flat = torch.zeros(input_ids.numel(), dtype=torch.float32, device=input_ids.device)
    disc_labels_flat.scatter_(0, masked_idx, replaced)
    disc_labels = disc_labels_flat.view_as(input_ids)

    # Active token mask logic mirrors DebertaV3RTDPretrainer.forward:
    # discriminator supervision applies to all active (non-padding) tokens.
    # Import helper directly to avoid dragging training runtime deps into this audit.
    from deberta.modeling.rtd import attention_mask_to_active_tokens

    pad_token_id = int(getattr(disc_cfg, "pad_token_id", tokenizer.pad_token_id))
    disc_active = attention_mask_to_active_tokens(
        input_ids=input_ids, attention_mask=attention_mask, pad_token_id=pad_token_id
    )

    disc_active_f = disc_active.to(dtype=torch.float32)
    disc_token_count = disc_active_f.sum()
    disc_denom = disc_token_count.clamp(min=1.0)

    disc_loss_per_token = torch.nn.functional.binary_cross_entropy_with_logits(
        disc_logits.float(), disc_labels.float(), reduction="none"
    )
    disc_loss_ref = (disc_loss_per_token * disc_active_f).sum() / disc_denom

    if not torch.allclose(disc_loss_ref, out.disc_loss_raw, atol=1e-6, rtol=0):
        # Provide a small diff summary.
        max_abs = float((disc_loss_ref - out.disc_loss_raw).abs().item())
        return _fail(
            "rtd_loss_integrity",
            f"Discriminator loss mismatch: ref={float(disc_loss_ref):.6f} impl={float(out.disc_loss_raw):.6f} (abs diff={max_abs:.2e})",
            hint="This usually means the discriminator active token-set is wrong (padding/attention mask handling) or reduction differs.",
        )

    # Gradient isolation (embedding_sharing='none'):
    model.zero_grad(set_to_none=True)
    out.disc_loss_raw.backward(retain_graph=True)
    gen_grads = [p.grad for p in model.generator.parameters() if p.requires_grad]
    if any(g is not None and float(g.abs().max().item()) > 0 for g in gen_grads):
        return _fail(
            "rtd_loss_integrity",
            "Discriminator loss produced gradients in generator parameters with embedding_sharing='none'.",
            hint="Corruption/sampling boundary must be stop-gradient (no grad through generator to discriminator).",
        )

    model.zero_grad(set_to_none=True)
    out.gen_loss_raw.backward(retain_graph=True)
    disc_grads = [p.grad for p in model.discriminator.parameters() if p.requires_grad]
    if any(g is not None and float(g.abs().max().item()) > 0 for g in disc_grads):
        return _fail(
            "rtd_loss_integrity",
            "Generator loss produced gradients in discriminator parameters.",
            hint="Generator MLM loss must only update generator + MLM head parameters.",
        )

    # Stochasticity sanity: gumbel sampling should not collapse to argmax on uniform logits.
    with torch.no_grad():
        logits0 = torch.zeros((64, tokenizer.vocab_size), dtype=torch.float32)
        samples = []
        for k in range(5):
            _seed_everything(100 + k)
            s = orig_gumbel(logits0, temperature=1.0, forbidden_vocab_mask=None)
            samples.append(s)
        uniq = torch.unique(torch.cat(samples))
        if int(uniq.numel()) <= 1:
            return _fail(
                "rtd_loss_integrity",
                "Gumbel sampling appears deterministic (only one unique token sampled on uniform logits).",
                hint="Sampling must be stochastic; do not replace it with argmax or deterministic substitution.",
            )

    # Forbidden-token sampling sanity: special ids must never be sampled when forbidden_vocab_mask is provided.
    with torch.no_grad():
        forb = getattr(model, "_forbidden_sample_token_mask", None)
        if isinstance(forb, torch.Tensor) and forb.numel() == tokenizer.vocab_size:
            logits0 = torch.zeros((256, tokenizer.vocab_size), dtype=torch.float32)
            _seed_everything(999)
            s = orig_gumbel(logits0, temperature=1.0, forbidden_vocab_mask=forb)
            bad = [
                tid
                for tid in tokenizer.all_special_ids
                if int(tid) < tokenizer.vocab_size and bool((s == int(tid)).any().item())
            ]
            if bad:
                return _fail(
                    "rtd_loss_integrity",
                    f"Forbidden special token ids were sampled: {bad}",
                    hint="Forbidden vocab mask must be applied before sampling (masked_fill to -inf/-1e9 is typical).",
                )

    # Token-set sanity: discriminator active set must include all non-padding active tokens.
    # Original non-padding specials (for example CLS/SEP) are valid all-negative targets.
    # Padding must never be active.
    pad_id = int(tokenizer.pad_token_id)
    active_pad = orig_padded_ids.eq(pad_id) & disc_active
    if bool(active_pad.any().item()):
        return _fail(
            "rtd_loss_integrity",
            "Discriminator active token set incorrectly includes padding positions "
            f"{active_pad.nonzero(as_tuple=False)[:5].tolist()}",
            hint="Discriminator supervision must exclude padding tokens.",
        )

    non_pad_special = torch.zeros_like(orig_padded_ids, dtype=torch.bool)
    for sid in tokenizer.all_special_ids:
        sid_i = int(sid)
        if sid_i == pad_id:
            continue
        non_pad_special = non_pad_special | orig_padded_ids.eq(sid_i)

    inactive_non_pad_special = non_pad_special & (~disc_active)
    if bool(inactive_non_pad_special.any().item()):
        return _fail(
            "rtd_loss_integrity",
            "Discriminator active token set dropped non-padding special-token positions "
            f"{inactive_non_pad_special.nonzero(as_tuple=False)[:5].tolist()}",
            hint="RTD discriminator loss should supervise all active non-padding tokens, including CLS/SEP.",
        )

    details = "Recomputed gen/disc losses match implementation; gradient boundaries OK; sampling stochastic; forbidden ids respected."
    if verbose:
        details += f"\n  gen_loss={float(out.gen_loss_raw):.6f} disc_loss={float(out.disc_loss_raw):.6f} total={float(out.loss):.6f}"
    return _pass("rtd_loss_integrity", details=details)


def check_checkpoint_roundtrip(repo_root: Path) -> CheckResult:
    ok, err = _try_import_transformers()
    if not ok:
        return _skip(
            "checkpoint_roundtrip", f"transformers not available ({err}); skipping model save/load check."
        )

    try:
        from deberta.modeling import DebertaRoPEConfig, DebertaRoPEModel
    except Exception as e:
        return _fail("checkpoint_roundtrip", f"Import failed: {type(e).__name__}: {e}")

    tokenizer = TinyTokenizer(vocab_size=128)
    cfg = DebertaRoPEConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        hidden_act="gelu",
        ffn_type="swiglu",
        use_bias=False,
        max_position_embeddings=64,
        type_vocab_size=2,
        norm_arch="post",
        attention_implementation="sdpa",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        norm_eps=1e-6,
        pad_token_id=tokenizer.pad_token_id,
        cls_token_id=tokenizer.cls_token_id,
        sep_token_id=tokenizer.sep_token_id,
        mask_token_id=tokenizer.mask_token_id,
    )

    _seed_everything(0)
    m1 = DebertaRoPEModel(cfg).eval()
    # Make deterministic input with padding.
    input_ids = torch.tensor(
        [
            [tokenizer.cls_token_id, 5, 6, 7, tokenizer.sep_token_id, tokenizer.pad_token_id],
            [
                tokenizer.cls_token_id,
                8,
                9,
                tokenizer.sep_token_id,
                tokenizer.pad_token_id,
                tokenizer.pad_token_id,
            ],
        ],
        dtype=torch.long,
    )
    attn = (input_ids != tokenizer.pad_token_id).to(torch.long)

    with torch.no_grad():
        y1 = m1(input_ids=input_ids, attention_mask=attn, return_dict=True).last_hidden_state.detach().clone()

    sd = m1.state_dict()
    m2 = DebertaRoPEModel(cfg).eval()
    missing = m2.load_state_dict(sd, strict=True)
    if getattr(missing, "missing_keys", []) or getattr(missing, "unexpected_keys", []):
        return _fail("checkpoint_roundtrip", f"Strict load had missing/unexpected keys: {missing}")

    with torch.no_grad():
        y2 = m2(input_ids=input_ids, attention_mask=attn, return_dict=True).last_hidden_state.detach().clone()

    if not torch.equal(y1, y2):
        max_abs = float((y1 - y2).abs().max().item())
        return _fail(
            "checkpoint_roundtrip",
            f"Model outputs differ after state_dict roundtrip (max|diff|={max_abs:.2e}).",
            hint="Checkpoint load must reproduce identical outputs in eval mode for deterministic inputs.",
        )

    return _pass("checkpoint_roundtrip")


def scan_unused_config_fields(repo_root: Path) -> CheckResult:
    """
    Lightweight config-to-behavior fidelity scan:
    - Parse config dataclasses (ModelConfig/DataConfig/TrainConfig) fields.
    - Grep the repo for occurrences outside config.py.
    Fields with 0 matches are suspicious (often 'parsed but never used').

    This is heuristic; it will produce some false positives/negatives, so it's WARN-level.
    """
    cfg_path = repo_root / "src" / "deberta" / "config.py"
    src = _read_text(cfg_path)
    tree = ast.parse(src, filename=str(cfg_path))

    dataclass_names = {"ModelConfig", "DataConfig", "TrainConfig"}
    fields: dict[str, list[str]] = {k: [] for k in dataclass_names}

    class Visitor(ast.NodeVisitor):
        def visit_ClassDef(self, node: ast.ClassDef) -> Any:
            if node.name not in dataclass_names:
                return
            # Collect AnnAssign targets at class scope.
            for stmt in node.body:
                if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                    fields[node.name].append(stmt.target.id)

    Visitor().visit(tree)

    all_fields = sorted({f for lst in fields.values() for f in lst})
    if not all_fields:
        return _warn("config_usage_scan", "Could not find dataclass fields in src/deberta/config.py.")

    # Search all .py files excluding config.py for field token.
    py_files = [p for p in repo_root.rglob("*.py") if p.is_file() and p.name != "config.py"]
    corpus = "\n".join(_read_text(p) for p in py_files)

    unused = []
    for f in all_fields:
        # crude token search
        if re.search(rf"\b{re.escape(f)}\b", corpus) is None:
            unused.append(f)

    if unused:
        return _warn(
            "config_usage_scan",
            "Config fields that appear unused (0 token matches outside config.py):\n"
            + "\n".join(f"  - {u}" for u in unused[:50])
            + ("\n  ... (truncated)" if len(unused) > 50 else ""),
            hint="Verify each listed field is either intentionally unused (deprecated) or wire it into behavior. "
            "In CI, treat new unused fields as failures.",
        )

    return _pass("config_usage_scan", "No obvious unused config fields found (heuristic scan).")


# -----------------------------
# Main
# -----------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="audit_contracts.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Contract-level audit harness for encoder modeling + pretraining pipelines.",
    )
    p.add_argument("--strict", action="store_true", help="Fail on WARN and SKIP (CI mode).")
    p.add_argument("--verbose", action="store_true", help="Print details for PASS results as well.")
    p.add_argument("--no-config-scan", action="store_true", help="Disable heuristic unused-config scan.")
    args = p.parse_args(argv)

    repo_root = _repo_root_from_script()
    # Ensure src/ is importable.
    sys.path.insert(0, str(repo_root / "src"))

    results: list[CheckResult] = []
    results.append(check_repo_layout(repo_root))
    # Static checks (no deps)
    results.append(check_export_cli_ignores_missing_keys(repo_root))
    results.append(check_collator_whole_word_fallback(repo_root))
    results.append(check_collator_determinism(repo_root))
    results.append(check_optimizer_state_ordering_risk(repo_root))
    results.append(check_doc_block_mask_contract(repo_root))
    results.append(check_attention_mask_to_active_tokens_contract(repo_root))
    results.append(check_rope_ffn_gating(repo_root))

    # Dynamic checks (require transformers)
    results.append(check_rope_layer_norm_order(repo_root))
    results.append(check_rope_attention_mask_leak(repo_root))
    results.append(check_rtd_loss_integrity(repo_root, verbose=bool(args.verbose)))
    results.append(check_checkpoint_roundtrip(repo_root))

    if not args.no_config_scan:
        results.append(scan_unused_config_fields(repo_root))

    # Print results
    results_sorted = sorted(results, key=lambda r: r.sort_key())
    for r in results_sorted:
        _print_result(r, verbose=bool(args.verbose))

    # Decide exit code
    fail = [r for r in results if r.status == "FAIL"]
    warn = [r for r in results if r.status == "WARN"]
    skip = [r for r in results if r.status == "SKIP"]

    print("")
    print(
        f"Summary: {len([r for r in results if r.status == 'PASS'])} PASS, "
        f"{len(warn)} WARN, {len(skip)} SKIP, {len(fail)} FAIL"
    )

    if fail:
        return 2
    if args.strict and (warn or skip):
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
