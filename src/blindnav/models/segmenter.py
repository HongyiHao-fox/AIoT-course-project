"""YOLO26 semantic segmentation: sidewalk / road layout."""
from __future__ import annotations

import cv2
import numpy as np

from blindnav.models.base import LazyModel
from blindnav.schemas import Segmentation


class SceneSegmenter(LazyModel):
    name = "YOLO26 semantic segmenter"

    def __init__(self, weights: str, *, device: str, imgsz: int = 1024) -> None:
        super().__init__()
        self.weights = weights
        self.device = device
        self.imgsz = imgsz

    def _build(self):
        from ultralytics import YOLO

        return YOLO(self.weights)

    def segment(self, image: np.ndarray) -> Segmentation:
        result = self.load().predict(
            source=image, device=self.device, imgsz=self.imgsz, verbose=False
        )[0]
        if getattr(result, "semantic_mask", None) is None:
            raise RuntimeError(
                "Model returned no semantic_mask: use semantic-segmentation weights "
                "(e.g. yolo26n-sem.pt) and an up-to-date ultralytics."
            )
        class_map = np.asarray(result.semantic_mask.data.cpu().numpy()).squeeze()
        h, w = image.shape[:2]
        if class_map.shape != (h, w):
            class_map = cv2.resize(
                class_map.astype(np.uint16), (w, h), interpolation=cv2.INTER_NEAREST
            )
        class_ids = {str(name).strip().lower(): int(idx) for idx, name in result.names.items()}
        return Segmentation(class_map=class_map.astype(np.int32), class_ids=class_ids)
