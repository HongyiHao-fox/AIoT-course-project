"""Drawing helpers for debug images."""
from __future__ import annotations

import cv2
import numpy as np

from blindnav.fusion import depth_range
from blindnav.schemas import Detection

# BGR colors
ZONE_COLORS = {"sidewalk": (0, 200, 0), "road": (255, 110, 0)}
BOX_COLOR = (0, 0, 255)


def colorize_depth(depth: np.ndarray) -> np.ndarray:
    low, span = depth_range(depth)
    norm = np.clip((depth - low) / span, 0, 1)
    norm[~np.isfinite(depth)] = 0
    return cv2.applyColorMap(np.uint8(norm * 255), cv2.COLORMAP_INFERNO)


def overlay_masks(image: np.ndarray, masks: dict[str, np.ndarray], alpha: float = 0.5) -> np.ndarray:
    out = image.copy()
    for name, mask in masks.items():
        sel = mask > 0
        if sel.any():
            color = np.array(ZONE_COLORS.get(name, (200, 200, 200)), dtype=np.float32)
            out[sel] = ((1 - alpha) * image[sel] + alpha * color).astype(np.uint8)
    return out


def _put_label(image: np.ndarray, text: str, origin: tuple[int, int]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, text, origin, font, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, text, origin, font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def draw_detections(image: np.ndarray, detections: list[Detection], show_depth: bool = True) -> np.ndarray:
    out = image.copy()
    for det in detections:
        x1, y1, x2, y2 = det.bbox_xyxy
        cv2.rectangle(out, (x1, y1), (x2, y2), BOX_COLOR, 2)
        label = f"{det.label} {det.confidence:.2f}"
        if show_depth and det.relative_closeness is not None:
            label += f" near={det.relative_closeness:.2f}"
        _put_label(out, label, (x1, max(18, y1 - 6)))
    return out
