import json

import pytest

from blindnav.export import save_outputs
from blindnav.prompts import build_scene_prompt


def test_analyze_runs_all_stages(navigator, image):
    result = navigator.analyze(image)
    assert [d.label for d in result.detections] == ["person", "car"]
    assert result.detections[1].relative_closeness > result.detections[0].relative_closeness
    assert result.zones["sidewalk"]["left"] == 100.0
    assert result.zones["road"]["right"] == 100.0
    assert result.description == "前方左侧有行人。"
    assert set(result.timings_ms) == {"detection", "depth", "segmentation", "description"}
    prompt = navigator.describer.prompts[0]
    assert "person" in prompt and "区域统计" in prompt


def test_analyze_without_description(navigator, image):
    result = navigator.analyze(image, describe=False)
    assert result.description is None
    assert navigator.describer.prompts == []


def test_describe_requires_vlm(navigator, image):
    navigator.describer = None
    with pytest.raises(RuntimeError):
        navigator.analyze(image, describe=True)


def test_low_vram_mode_unloads_models(navigator, image):
    navigator.keep_loaded = False
    navigator.analyze(image)
    assert navigator.detector.unloaded == 1


def test_save_outputs_writes_expected_files(navigator, image, tmp_path):
    result = navigator.analyze(image)
    files = {p.name for p in save_outputs(result, image, tmp_path, source="test.jpg")}
    assert files == {
        "detected.jpg", "depth.jpg", "depth.npy", "sidewalk_mask.png", "road_mask.png",
        "semantic_overlay.jpg", "fusion.jpg", "analysis.json", "report.txt",
    }
    data = json.loads((tmp_path / "analysis.json").read_text(encoding="utf-8"))
    assert data["image"] == "test.jpg"
    assert len(data["detections"]) == 2
    assert len(data["notes"]) == 2


def test_prompt_without_zones_uses_detection_prompt(detections):
    prompt = build_scene_prompt(detections, zones=None, max_objects=1)
    assert "person" in prompt and "car" not in prompt
    assert "区域统计" not in prompt
