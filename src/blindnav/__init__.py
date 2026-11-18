"""blindnav: perception pipeline for a wearable navigation aid for blind users.

Detects obstacles, estimates relative depth, segments sidewalk/road and
describes the scene (including signboards) with a vision-language model.
"""
from blindnav.config import Settings, load_settings
from blindnav.pipeline import Navigator
from blindnav.schemas import AnalysisResult, Detection, Segmentation

__version__ = "0.1.0"

__all__ = [
    "AnalysisResult",
    "Detection",
    "Navigator",
    "Segmentation",
    "Settings",
    "load_settings",
]
