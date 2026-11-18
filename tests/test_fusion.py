import numpy as np
import pytest

from blindnav.fusion import attach_depth, depth_range, zone_coverage


def test_attach_depth_ranks_nearer_object_higher(detections):
    depth = np.tile(np.linspace(0, 10, 300, dtype=np.float32), (120, 1))
    attach_depth(detections, depth)
    person, car = detections
    assert car.relative_closeness > person.relative_closeness
    assert 0 <= person.relative_closeness <= 1
    assert car.relative_depth is not None


def test_attach_depth_ignores_nan_pixels(detections):
    depth = np.ones((120, 300), np.float32)
    depth[:, :150] = np.nan
    attach_depth(detections, depth)
    assert detections[0].relative_depth is None  # left box is all NaN
    assert detections[1].relative_depth == pytest.approx(1.0)


def test_depth_range_requires_finite_values():
    with pytest.raises(ValueError):
        depth_range(np.full((4, 4), np.nan))


def test_zone_coverage_thirds():
    mask = np.zeros((10, 30), np.uint8)
    mask[:, :10] = 255
    zones = zone_coverage(mask)
    assert zones == {"image": 33.3, "left": 100.0, "center": 0.0, "right": 0.0}
