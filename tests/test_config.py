from pathlib import Path

import pytest

from blindnav.config import MODEL_ROOT_ENV, Settings, load_settings

DEFAULT_YAML = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


def test_default_yaml_matches_defaults(monkeypatch):
    monkeypatch.delenv(MODEL_ROOT_ENV, raising=False)
    assert load_settings(DEFAULT_YAML).to_dict() == Settings().to_dict()


def test_yaml_overrides_nested_values(tmp_path, monkeypatch):
    monkeypatch.delenv(MODEL_ROOT_ENV, raising=False)
    cfg = tmp_path / "c.yaml"
    cfg.write_text("model_root: /models\ndetector:\n  conf: 0.5\n")
    s = load_settings(cfg)
    assert s.model_root == Path("/models")
    assert s.detector.conf == 0.5
    assert s.detector.imgsz == 640


def test_unknown_key_is_rejected(tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text("detector:\n  confidence: 0.5\n")
    with pytest.raises(ValueError, match="detector.confidence"):
        load_settings(cfg)


def test_env_overrides_model_root(monkeypatch):
    monkeypatch.setenv(MODEL_ROOT_ENV, "/data/models")
    assert load_settings().model_root == Path("/data/models")


def test_resolve_and_weights(tmp_path):
    s = Settings(model_root=tmp_path)
    assert s.resolve("x/y") == tmp_path / "x/y"
    assert s.resolve("/abs/p") == Path("/abs/p")
    assert s.weights("yolo26s.pt") == "yolo26s.pt"  # missing -> let Ultralytics download
    (tmp_path / "yolo26s.pt").write_bytes(b"")
    assert s.weights("yolo26s.pt") == str(tmp_path / "yolo26s.pt")
