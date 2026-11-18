"""Combine detections with depth and segmentation results."""
from __future__ import annotations

import numpy as np

from blindnav.schemas import Detection
from blindnav.utils.geometry import inner_box


def depth_range(depth: np.ndarray, low_pct: float = 5, high_pct: float = 95) -> tuple[float, float]:
    """Robust (low, span) of a relative depth map, ignoring non-finite values."""
    finite = np.isfinite(depth)
    if not finite.any():
        raise ValueError("Depth map contains no finite values")
    low, high = np.percentile(depth[finite], [low_pct, high_pct])
    return float(low), max(float(high - low), 1e-6)


def attach_depth(detections: list[Detection], depth: np.ndarray, roi_fraction: float = 0.5) -> list[Detection]:
    """Set relative_depth (median in the box core) and relative_closeness (0..1) in place."""
    low, span = depth_range(depth)
    for det in detections:
        x1, y1, x2, y2 = inner_box(det.bbox_xyxy, roi_fraction)
        patch = depth[y1:y2, x1:x2]
        valid = patch[np.isfinite(patch)]
        if valid.size:
            score = float(np.median(valid))
            det.relative_depth = round(score, 4)
            det.relative_closeness = round(float(np.clip((score - low) / span, 0, 1)), 4)
    return detections


def zone_coverage(mask: np.ndarray) -> dict[str, float]:
    """Percent of the whole image and of its left/center/right thirds covered by a mask."""
    w = mask.shape[1]

    def pct(region: np.ndarray) -> float:
        return round(float(np.count_nonzero(region)) / max(1, region.size) * 100, 1)

    return {
        "image": pct(mask),
        "left": pct(mask[:, : w // 3]),
        "center": pct(mask[:, w // 3 : 2 * w // 3]),
        "right": pct(mask[:, 2 * w // 3 :]),
    }
