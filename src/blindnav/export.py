"""Save pipeline results (images + JSON) to a folder."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from blindnav.pipeline import ZONE_CLASSES
from blindnav.schemas import AnalysisResult
from blindnav.visualize import colorize_depth, draw_detections, overlay_masks


def save_outputs(
    result: AnalysisResult,
    image: np.ndarray,
    out_dir: str | Path,
    source: str | Path | None = None,
) -> list[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    def write_image(name: str, img: np.ndarray) -> None:
        path = out / name
        if not cv2.imwrite(str(path), img):
            raise OSError(f"Could not write {path}")
        written.append(path)

    write_image("detected.jpg", draw_detections(image, result.detections, show_depth=False))

    if result.depth is not None:
        write_image("depth.jpg", colorize_depth(result.depth))
        np.save(out / "depth.npy", result.depth)
        written.append(out / "depth.npy")

    base = image
    if result.segmentation is not None:
        masks = {name: result.segmentation.mask(name) for name in ZONE_CLASSES}
        for name, mask in masks.items():
            write_image(f"{name}_mask.png", mask)
        base = overlay_masks(image, masks)
        write_image("semantic_overlay.jpg", base)

    write_image("fusion.jpg", draw_detections(base, result.detections))

    payload = {"image": str(source) if source else None, **result.to_dict()}
    json_path = out / "analysis.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    written.append(json_path)

    if result.description:
        report = out / "report.txt"
        report.write_text(result.description, encoding="utf-8")
        written.append(report)
    return written
