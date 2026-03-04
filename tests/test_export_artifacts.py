from __future__ import annotations

import types
from pathlib import Path

from deberta.config import DataConfig, ModelConfig, TrainConfig
from deberta.modeling.export_utils import write_export_readme_and_license


def test_write_export_readme_rope_usage_warns_auto_model_limitation(tmp_path: Path) -> None:
    out_dir = tmp_path / "rope-export"
    out_dir.mkdir(parents=True, exist_ok=True)

    write_export_readme_and_license(
        out_dir,
        model_cfg=ModelConfig(backbone_type="rope"),
        data_cfg=DataConfig(max_seq_length=777),
        train_cfg=TrainConfig(max_steps=100),
        embedding_sharing="gdes",
    )

    text = (out_dir / "README.md").read_text(encoding="utf-8")
    assert "DebertaRoPEModel.from_pretrained" in text
    assert 'model = AutoModel.from_pretrained("path/to/this/dir")' not in text
    assert "model_type" in text
    assert "| Max sequence length | 777 |" in text
    assert (out_dir / "LICENSE").exists()


def test_write_export_readme_hf_uses_auto_model_snippet(tmp_path: Path) -> None:
    out_dir = tmp_path / "hf-export"
    out_dir.mkdir(parents=True, exist_ok=True)

    write_export_readme_and_license(
        out_dir,
        model_cfg=ModelConfig(backbone_type="hf_deberta_v2"),
        data_cfg=DataConfig(max_seq_length=333),
        train_cfg=TrainConfig(max_steps=100),
        embedding_sharing="gdes",
    )

    text = (out_dir / "README.md").read_text(encoding="utf-8")
    assert "AutoModel.from_pretrained" in text
    assert "DebertaRoPEModel.from_pretrained" not in text
    assert "| Max sequence length | 333 |" in text
    assert (out_dir / "LICENSE").exists()


def test_write_export_readme_uses_export_config_dimensions_when_available(tmp_path: Path) -> None:
    out_dir = tmp_path / "hf-export-effective-config"
    out_dir.mkdir(parents=True, exist_ok=True)

    export_cfg = types.SimpleNamespace(
        hidden_size=768,
        num_hidden_layers=6,
        num_attention_heads=12,
        max_position_embeddings=4096,
    )

    write_export_readme_and_license(
        out_dir,
        model_cfg=ModelConfig(backbone_type="hf_deberta_v2", hf_model_size="small"),
        export_config=export_cfg,
        data_cfg=None,
        train_cfg=TrainConfig(max_steps=100),
        embedding_sharing="gdes",
    )

    text = (out_dir / "README.md").read_text(encoding="utf-8")
    assert "# hf_deberta_v2-768h-6L-12H" in text
    assert "| Max sequence length | 4096 |" in text
    assert (out_dir / "LICENSE").exists()
