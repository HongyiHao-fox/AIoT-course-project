"""Runtime settings: model locations, thresholds and device.

Precedence (highest first): CLI flags > ``BLINDNAV_MODEL_ROOT`` env var >
YAML config file > defaults below.
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

MODEL_ROOT_ENV = "BLINDNAV_MODEL_ROOT"
DEFAULT_MODEL_ROOT = "/root/autodl-tmp"


@dataclass
class DetectorConfig:
    weights: str = "yolo26s.pt"
    conf: float = 0.25
    imgsz: int = 640


@dataclass
class DepthConfig:
    repo_dir: str = "Depth-Anything-V2"
    encoder: str = "vits"
    # Fraction of each box (centered) sampled for depth, to avoid background pixels.
    roi_fraction: float = 0.5


@dataclass
class SegmenterConfig:
    weights: str = "yolo26n-sem.pt"
    imgsz: int = 1024


@dataclass
class VLMConfig:
    model_dir: str = "Qwen3-VL-8B-Instruct"
    max_new_tokens: int = 320
    max_objects: int = 25


@dataclass
class Settings:
    model_root: Path = field(default_factory=lambda: Path(DEFAULT_MODEL_ROOT))
    device: str = "auto"
    # Horizontal position thresholds (fraction of image width).
    left_boundary: float = 0.4
    right_boundary: float = 0.6
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    depth: DepthConfig = field(default_factory=DepthConfig)
    segmenter: SegmenterConfig = field(default_factory=SegmenterConfig)
    vlm: VLMConfig = field(default_factory=VLMConfig)

    def resolve(self, path: str | Path) -> Path:
        """Resolve a path relative to ``model_root`` (absolute paths are kept)."""
        p = Path(path).expanduser()
        return p if p.is_absolute() else Path(self.model_root).expanduser() / p

    def weights(self, name: str) -> str:
        """Local weights under ``model_root`` if present, else the bare name
        so Ultralytics can download official weights automatically."""
        local = self.resolve(name)
        return str(local) if local.is_file() else str(name)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["model_root"] = str(self.model_root)
        return data


def _merge(target: Any, data: dict[str, Any], prefix: str = "") -> None:
    names = {f.name for f in fields(target)}
    for key, value in data.items():
        if key not in names:
            raise ValueError(f"Unknown setting: {prefix}{key}")
        current = getattr(target, key)
        if is_dataclass(current):
            if not isinstance(value, dict):
                raise ValueError(f"Setting {prefix}{key} must be a mapping")
            _merge(current, value, f"{prefix}{key}.")
        elif key == "model_root":
            setattr(target, key, Path(value))
        else:
            setattr(target, key, value)


def load_settings(path: str | Path | None = None) -> Settings:
    """Build settings from defaults, an optional YAML file and the environment."""
    settings = Settings()
    if path is not None:
        import yaml

        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        _merge(settings, data)
    env_root = os.getenv(MODEL_ROOT_ENV)
    if env_root:
        settings.model_root = Path(env_root)
    return settings
