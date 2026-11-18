"""Model wrappers. Each loads its weights lazily on first use."""
from blindnav.models.depth import DepthEstimator
from blindnav.models.detector import ObjectDetector
from blindnav.models.segmenter import SceneSegmenter
from blindnav.models.vlm import SceneDescriber

__all__ = ["DepthEstimator", "ObjectDetector", "SceneDescriber", "SceneSegmenter"]
