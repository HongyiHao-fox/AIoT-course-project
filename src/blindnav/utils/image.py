"""Image loading and conversion. All arrays are OpenCV BGR uint8."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def load_image(path: str | Path) -> np.ndarray:
    src = Path(path).expanduser().resolve()
    if not src.is_file():
        raise FileNotFoundError(f"Image not found: {src}")
    image = cv2.imread(str(src))
    if image is None:
        raise ValueError(f"OpenCV cannot decode image: {src}")
    return image


def decode_image(data: bytes) -> np.ndarray:
    """Decode JPEG/PNG bytes (e.g. an HTTP upload)."""
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Data is not a decodable image")
    return image


def bgr_to_pil(image: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
