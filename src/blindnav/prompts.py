"""Prompts for Qwen3-VL (Chinese output for end users)."""
from __future__ import annotations

import json

from blindnav.schemas import Detection

CHAT_SYSTEM_PROMPT = """你是一个用于盲人辅助导航原型研究的视觉分析助手。请用简洁中文回答。
当用户提供街景图片时，优先分析：路牌或告示文字、施工围挡、台阶、路缘、障碍物、行人信号等，指出图中相对方位。
清楚区分可直接看见的信息和不确定的推测；看不清文字就说看不清。
不要仅凭图片判断可安全通行、给出精确距离或指示用户过马路；真实避障需要独立测距与安全模块。"""

CHAT_IMAGE_REQUEST = "请分析图片中的重要路牌文字、障碍物、相对位置和潜在危险，并明确不确定的信息。"

_SCENE_DETECTIONS_ONLY = (
    "你是用于研究演示的盲人出行环境描述助手。结合输入图片与以下YOLO检测结果，"
    "简洁地用中文描述：前方可见的主要物体、它们大致位于画面的左中右、需要关注的可见环境特征。"
    "不要断言道路安全、不要给过街许可、不要给出距离米数。"
    "检测结果并非绝对正确；当图片不足以支持判断时，请明确说明不确定。"
    "请输出最多5句话。"
)

_SCENE_FULL = (
    "你是一名研究原型的城市步行场景描述助手，为视障人士提供简洁的环境信息。\n"
    "你会看到原始街景图片和三种模型输出的结构化数据。"
    "请综合图片、目标检测类别位置、相对深度、道路/人行道语义分割占比，"
    "中文回答：(1)可见的障碍物与左右位置；(2)画面中的人行道和道路分布；"
    "(3)无法确定的风险及需要人工确认的事项。最多6句话。\n"
    "严格规则：相对深度分数不代表米数，不跨图片比较；不能从像素占比推断真实可走路径；"
    "禁止声称可以安全前进或允许过街；交通灯检测框不等于行人通行状态；"
    "若模型之间不一致，以不确定方式描述。"
)


def build_scene_prompt(
    detections: list[Detection],
    zones: dict[str, dict[str, float]] | None = None,
    max_objects: int = 25,
) -> str:
    objects = [
        {
            "class": d.label,
            "confidence": d.confidence,
            "position": d.position,
            "relative_closeness": d.relative_closeness,
        }
        for d in detections[:max_objects]
    ]
    data = "对象检测：" + json.dumps(objects, ensure_ascii=False)
    if zones is None:
        return f"{_SCENE_DETECTIONS_ONLY}\n{data}"
    return f"{_SCENE_FULL}\n{data}\n区域统计：{json.dumps(zones, ensure_ascii=False)}"
