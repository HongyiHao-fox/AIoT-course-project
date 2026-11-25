from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

from blindnav.models.base import LazyModel

ENCODER_CONFIGS = {
    "vits": {"features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"features": 256, "out_channels": [256, 512, 1024, 1024]},
}


class DepthEstimator(LazyModel):
    name = "Depth Anything V2"

    def __init__(self, repo_dir: str | Path, *, device: str, encoder: str = "vits") -> None:
        super().__init__()
        if encoder not in ENCODER_CONFIGS:
            raise ValueError(f"Unknown encoder {encoder!r}; choose from {sorted(ENCODER_CONFIGS)}")
        self.repo_dir = Path(repo_dir).expanduser().resolve()
        self.device = device
        self.encoder = encoder

    @property
    def checkpoint(self) -> Path:
        return self.repo_dir / "checkpoints" / f"depth_anything_v2_{self.encoder}.pth"

    def _build(self):
        if not self.checkpoint.is_file():
            raise FileNotFoundError(f"Depth checkpoint not found: {self.checkpoint}")
        import torch

        # Depth-Anything-V2 is a cloned repo, not a pip package.
        if str(self.repo_dir) not in sys.path:
            sys.path.insert(0, str(self.repo_dir))
        from depth_anything_v2.dpt import DepthAnythingV2

        model = DepthAnythingV2(encoder=self.encoder, **ENCODER_CONFIGS[self.encoder])
        state = torch.load(str(self.checkpoint), map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        return model.to(self.device).eval()

    def estimate(self, image: np.ndarray) -> np.ndarray:
        """HxW float32 relative depth; higher usually means nearer."""
        import torch

        with torch.inference_mode():
            raw = self.load().infer_image(image)
        depth = np.asarray(raw, dtype=np.float32)
        h, w = image.shape[:2]
        if depth.shape != (h, w):
            depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)
        return depth
