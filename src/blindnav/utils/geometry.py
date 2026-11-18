"""Bounding-box helpers."""
from __future__ import annotations

from collections.abc import Sequence

Box = tuple[int, int, int, int]


def clip_box(x1: int, y1: int, x2: int, y2: int, width: int, height: int) -> Box | None:
    """Clip a box to the image; None if it becomes empty."""
    x1, x2 = max(0, min(x1, width)), max(0, min(x2, width))
    y1, y2 = max(0, min(y1, height)), max(0, min(y2, height))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def horizontal_position(center_x: float, width: int, left: float = 0.4, right: float = 0.6) -> str:
    """Return 'left', 'center' or 'right' for a horizontal pixel position."""
    ratio = center_x / width
    if ratio < left:
        return "left"
    if ratio > right:
        return "right"
    return "center"


def inner_box(box: Sequence[int], fraction: float = 0.5) -> Box:
    """Centered sub-box covering ``fraction`` of the width and height."""
    x1, y1, x2, y2 = box
    mx = int((x2 - x1) * (1 - fraction) / 2)
    my = int((y2 - y1) * (1 - fraction) / 2)
    return x1 + mx, y1 + my, x2 - mx, y2 - my
