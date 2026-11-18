"""YOLO26 object detection: obstacles, people, vehicles, traffic lights."""
from __future__ import annotations

import numpy as np

from blindnav.models.base import LazyModel
from blindnav.schemas import Detection
from blindnav.utils.geometry import clip_box, horizontal_position


class ObjectDetector(LazyModel):
    name = "YOLO26 detector"

    def __init__(
        self,
        weights: str,
        *,
        device: str,
        conf: float = 0.25,
        imgsz: int = 640,
        left_boundary: float = 0.4,
        right_boundary: float = 0.6,
    ) -> None:
        super().__init__()
        self.weights = weights
        self.device = device
        self.conf = conf
        self.imgsz = imgsz
        self.left_boundary = left_boundary
        self.right_boundary = right_boundary

    def _build(self):
        from ultralytics import YOLO

        return YOLO(self.weights)

    def detect(self, image: np.ndarray) -> list[Detection]:
        h, w = image.shape[:2]
        result = self.load().predict(
            source=image, device=self.device, conf=self.conf, imgsz=self.imgsz, verbose=False
        )[0]
        detections: list[Detection] = []
        if result.boxes is None:
            return detections
        for box in result.boxes:
            coords = [int(round(float(v))) for v in box.xyxy[0].tolist()]
            clipped = clip_box(*coords, w, h)
            if clipped is None:
                continue
            x1, _, x2, _ = clipped
            detections.append(
                Detection(
                    label=str(result.names[int(box.cls[0])]),
                    confidence=round(float(box.conf[0]), 3),
                    bbox_xyxy=list(clipped),
                    position=horizontal_position(
                        (x1 + x2) / 2, w, self.left_boundary, self.right_boundary
                    ),
                )
            )
        return detections
