import pytest

from blindnav.utils.geometry import clip_box, horizontal_position, inner_box


@pytest.mark.parametrize(
    ("center_x", "expected"),
    [(10, "left"), (39, "left"), (40, "center"), (50, "center"), (60, "center"), (61, "right")],
)
def test_horizontal_position(center_x, expected):
    assert horizontal_position(center_x, 100) == expected


def test_clip_box_clips_to_image():
    assert clip_box(-5, -5, 120, 50, 100, 80) == (0, 0, 100, 50)


def test_clip_box_rejects_empty():
    assert clip_box(50, 10, 50, 40, 100, 100) is None
    assert clip_box(120, 10, 150, 40, 100, 100) is None


def test_inner_box_half():
    assert inner_box((0, 0, 100, 40), 0.5) == (25, 10, 75, 30)
