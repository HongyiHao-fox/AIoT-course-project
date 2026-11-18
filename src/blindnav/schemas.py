"""Data structures passed between pipeline stages."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

DEPTH_NOTE = (
    "Depth Anything V2 outputs relative depth only. Higher scores usually mean nearer. "
    "relative_closeness is normalized within this image: it is not a distance in meters "
    "and must not be used as a safe-clearance estimate."
)
SEGMENTATION_NOTE = "Sidewalk/road labels describe visual appearance, not verified safe passage."


@dataclass
class Detection:
    label: str
    confidence: float
    bbox_xyxy: list[int]
    position: str  # "left" | "center" | "right"
    relative_depth: float | None = None
    relative_closeness: float | None = None  # 0..1 within this image, higher = nearer

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Segmentation:
    class_map: np.ndarray  # HxW class ids
    class_ids: dict[str, int]  # lowercase class name -> id

    def mask(self, name: str) -> np.ndarray:
        """uint8 mask (0/255) for a class; all zeros if the model lacks it."""
        idx = self.class_ids.get(name.lower())
        if idx is None:
            return np.zeros(self.class_map.shape[:2], np.uint8)
        return np.uint8(self.class_map == idx) * 255


@dataclass
class AnalysisResult:
    image_size_wh: tuple[int, int]
    detections: list[Detection] = field(default_factory=list)
    zones: dict[str, dict[str, float]] | None = None
    description: str | None = None
    timings_ms: dict[str, float] = field(default_factory=dict)
    # Raw arrays for visualization; not included in to_dict().
    depth: np.ndarray | None = field(default=None, repr=False)
    segmentation: Segmentation | None = field(default=None, repr=False)

    def to_dict(self) -> dict[str, Any]:
        notes = []
        if self.depth is not None:
            notes.append(DEPTH_NOTE)
        if self.zones is not None:
            notes.append(SEGMENTATION_NOTE)
        return {
            "image_size_wh": list(self.image_size_wh),
            "detections": [d.to_dict() for d in self.detections],
            "zones": self.zones,
            "description": self.description,
            "timings_ms": self.timings_ms,
            "notes": notes,
        }
