"""The Navigator: runs detection -> depth -> segmentation -> VLM on one frame."""
from __future__ import annotations

import logging
import time
from contextlib import contextmanager

import numpy as np

from blindnav.config import Settings, load_settings
from blindnav.fusion import attach_depth, zone_coverage
from blindnav.models import DepthEstimator, ObjectDetector, SceneDescriber, SceneSegmenter
from blindnav.models.base import LazyModel
from blindnav.prompts import build_scene_prompt
from blindnav.schemas import AnalysisResult
from blindnav.utils.device import resolve_device
from blindnav.utils.image import bgr_to_pil

log = logging.getLogger(__name__)

ZONE_CLASSES = ("sidewalk", "road")


class Navigator:
    """Perception pipeline for one camera frame.

    keep_loaded=True keeps every model in memory (fast, for the server / live use).
    keep_loaded=False frees each model after its stage (fits smaller GPUs).
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        segmentation: bool = True,
        vlm: bool = False,
        keep_loaded: bool = True,
    ) -> None:
        s = settings or load_settings()
        self.settings = s
        self.keep_loaded = keep_loaded
        self.device = resolve_device(s.device)
        self.detector = ObjectDetector(
            s.weights(s.detector.weights),
            device=self.device,
            conf=s.detector.conf,
            imgsz=s.detector.imgsz,
            left_boundary=s.left_boundary,
            right_boundary=s.right_boundary,
        )
        self.depth = DepthEstimator(s.resolve(s.depth.repo_dir), device=self.device, encoder=s.depth.encoder)
        self.segmenter = (
            SceneSegmenter(s.weights(s.segmenter.weights), device=self.device, imgsz=s.segmenter.imgsz)
            if segmentation
            else None
        )
        self.describer = (
            SceneDescriber(s.resolve(s.vlm.model_dir), max_new_tokens=s.vlm.max_new_tokens) if vlm else None
        )

    def models(self) -> list[LazyModel]:
        return [m for m in (self.detector, self.depth, self.segmenter, self.describer) if m is not None]

    def warmup(self) -> None:
        for model in self.models():
            model.load()

    def close(self) -> None:
        for model in self.models():
            model.unload()

    @contextmanager
    def _stage(self, name: str, model: LazyModel, timings: dict[str, float]):
        log.info("Running %s on %s", model.name, self.device)
        start = time.perf_counter()
        try:
            yield
        finally:
            timings[name] = round((time.perf_counter() - start) * 1000, 1)
            if not self.keep_loaded:
                model.unload()

    def analyze(self, image: np.ndarray, *, describe: bool | None = None) -> AnalysisResult:
        """Analyze one BGR frame. ``describe`` defaults to True when the VLM is enabled."""
        if describe is None:
            describe = self.describer is not None
        if describe and self.describer is None:
            raise RuntimeError("Navigator was created with vlm=False")

        h, w = image.shape[:2]
        result = AnalysisResult(image_size_wh=(w, h))
        timings = result.timings_ms

        with self._stage("detection", self.detector, timings):
            result.detections = self.detector.detect(image)

        with self._stage("depth", self.depth, timings):
            result.depth = self.depth.estimate(image)
        attach_depth(result.detections, result.depth, self.settings.depth.roi_fraction)

        if self.segmenter is not None:
            with self._stage("segmentation", self.segmenter, timings):
                result.segmentation = self.segmenter.segment(image)
            result.zones = {name: zone_coverage(result.segmentation.mask(name)) for name in ZONE_CLASSES}

        if describe:
            prompt = build_scene_prompt(result.detections, result.zones, self.settings.vlm.max_objects)
            with self._stage("description", self.describer, timings):
                result.description = self.describer.describe(bgr_to_pil(image), prompt)

        return result
