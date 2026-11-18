from __future__ import annotations

import numpy as np
import pytest

from blindnav.schemas import Detection, Segmentation


class FakeDetector:
    name = "fake detector"

    def __init__(self, detections):
        self.detections = detections
        self.unloaded = 0

    def detect(self, image):
        return [Detection(**vars(d)) for d in self.detections]

    def unload(self):
        self.unloaded += 1


class FakeDepth:
    name = "fake depth"

    def estimate(self, image):
        h, w = image.shape[:2]
        # Nearer (higher) toward the right side of the image.
        return np.tile(np.linspace(0, 10, w, dtype=np.float32), (h, 1))

    def unload(self):
        pass


class FakeSegmenter:
    name = "fake segmenter"

    def segment(self, image):
        h, w = image.shape[:2]
        class_map = np.zeros((h, w), np.int32)
        class_map[:, : w // 3] = 1  # sidewalk on the left third
        class_map[:, 2 * w // 3 :] = 2  # road on the right third
        return Segmentation(class_map=class_map, class_ids={"background": 0, "sidewalk": 1, "road": 2})

    def unload(self):
        pass


class FakeDescriber:
    name = "fake vlm"

    def __init__(self):
        self.prompts = []

    def describe(self, image, prompt):
        self.prompts.append(prompt)
        return "前方左侧有行人。"

    def unload(self):
        pass


@pytest.fixture
def image():
    return np.full((120, 300, 3), 127, np.uint8)


@pytest.fixture
def detections():
    return [
        Detection(label="person", confidence=0.9, bbox_xyxy=[10, 10, 60, 100], position="left"),
        Detection(label="car", confidence=0.8, bbox_xyxy=[220, 20, 290, 110], position="right"),
    ]


@pytest.fixture
def navigator(detections):
    from blindnav.config import Settings
    from blindnav.pipeline import Navigator

    nav = Navigator(Settings(device="cpu"), segmentation=True, vlm=True)
    nav.detector = FakeDetector(detections)
    nav.depth = FakeDepth()
    nav.segmenter = FakeSegmenter()
    nav.describer = FakeDescriber()
    return nav
