from __future__ import annotations

from pathlib import Path

import pytest
from _config_factories import make_data_config, make_model_config, make_train_config
from _fakes import BackboneConfigStub

from deberta.modeling.export_utils import write_export_readme_and_license


@pytest.mark.parametrize(
    ("backbone_type", "max_seq_length", "required", "forbidden", "extra_required"),
    [
        (
            "rope",
            777,
            "DebertaRoPEModel.from_pretrained",
            'model = AutoModel.from_pretrained("path/to/this/dir")',
            "model_type",
        ),
        ("hf_deberta_v2", 333, "AutoModel.from_pretrained", "DebertaRoPEModel.from_pretrained", None),
    ],
)
def test_write_export_readme_uses_backbone_specific_loading_snippet(
    tmp_path: Path,
    backbone_type: str,
    max_seq_length: int,
    required: str,
    forbidden: str,
    extra_required: str | None,
) -> None:
    out_dir = tmp_path / f"{backbone_type}-export"
    out_dir.mkdir(parents=True, exist_ok=True)

    write_export_readme_and_license(
        out_dir,
        model_cfg=make_model_config(backbone_type=backbone_type),
        data_cfg=make_data_config(packing={"max_seq_length": max_seq_length}),
        train_cfg=make_train_config(max_steps=100),
        embedding_sharing="gdes",
    )

    text = (out_dir / "README.md").read_text(encoding="utf-8")
    assert required in text
    assert forbidden not in text
    if extra_required is not None:
        assert extra_required in text
    assert f"| Max sequence length | {max_seq_length} |" in text
    assert (out_dir / "LICENSE").exists()


def test_write_export_readme_uses_export_config_dimensions_when_available(tmp_path: Path) -> None:
    out_dir = tmp_path / "hf-export-effective-config"
    out_dir.mkdir(parents=True, exist_ok=True)

    export_cfg = BackboneConfigStub(
        hidden_size=768,
        num_hidden_layers=6,
        num_attention_heads=12,
        max_position_embeddings=4096,
    )

    write_export_readme_and_license(
        out_dir,
        model_cfg=make_model_config(backbone_type="hf_deberta_v2", hf={"model_size": "small"}),
        export_config=export_cfg,
        data_cfg=None,
        train_cfg=make_train_config(max_steps=100),
        embedding_sharing="gdes",
    )

    text = (out_dir / "README.md").read_text(encoding="utf-8")
    assert "# hf_deberta_v2-768h-6L-12H" in text
    assert "| Max sequence length | 4096 |" in text
    assert (out_dir / "LICENSE").exists()
