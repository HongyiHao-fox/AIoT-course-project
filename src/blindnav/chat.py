"""Interactive Qwen3-VL terminal chat against a running vLLM (OpenAI-compatible) server."""
from __future__ import annotations

import base64
import io
import os
from pathlib import Path

from blindnav.prompts import CHAT_IMAGE_REQUEST, CHAT_SYSTEM_PROMPT

HELP = "/image /path/to/photo.jpg 发送图片\n/clear 清空对话\n/exit 退出\n普通文字直接发送"


def image_data_uri(path_str: str) -> str:
    from PIL import Image, ImageOps

    path = Path(os.path.expanduser(path_str)).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"找不到图片：{path}")
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image.thumbnail((1280, 1280))
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def run_chat(
    base_url: str | None = None,
    model: str | None = None,
    max_tokens: int | None = None,
    max_turns: int | None = None,
) -> int:
    from openai import OpenAI

    base_url = base_url or os.getenv("VLM_BASE_URL", "http://127.0.0.1:8000/v1")
    model = model or os.getenv("VLM_MODEL", "")
    max_tokens = max_tokens or int(os.getenv("VLM_MAX_TOKENS", "512"))
    max_turns = max_turns or int(os.getenv("VLM_MAX_TURNS", "6"))

    client = OpenAI(base_url=base_url, api_key=os.getenv("VLM_API_KEY", "EMPTY"), timeout=180.0, max_retries=1)
    try:
        models = client.models.list().data
        if not models:
            raise RuntimeError("服务未返回任何模型")
        model = model or models[0].id
    except Exception as exc:
        print(f"无法连接 vLLM 服务：{base_url}\n原因：{exc}\n请先在另一终端启动 vllm serve。")
        return 1

    system = {"role": "system", "content": CHAT_SYSTEM_PROMPT}
    history: list[dict] = []
    print(f"已连接模型：{model}")
    print("直接输入文字聊天；/image 路径 发送图片；/clear 清空历史；/help 帮助；/exit 退出。")

    while True:
        try:
            user_input = input("\n你> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n已退出。")
            return 0
        if not user_input:
            continue
        cmd = user_input.lower()
        if cmd in ("/exit", "/quit", "exit", "quit"):
            print("已退出。")
            return 0
        if cmd == "/help":
            print(HELP)
            continue
        if cmd == "/clear":
            history = []
            print("对话历史已清空。")
            continue
        if cmd.startswith("/image "):
            path = user_input[7:].strip().strip("\"'")
            try:
                uri = image_data_uri(path)
            except Exception as exc:
                print(f"图片读取失败：{exc}")
                continue
            user_message = {
                "role": "user",
                "content": [
                    {"type": "text", "text": CHAT_IMAGE_REQUEST},
                    {"type": "image_url", "image_url": {"url": uri}},
                ],
            }
        else:
            user_message = {"role": "user", "content": user_input}

        # Keep only recent turns to bound context length.
        recent = history[-(max_turns * 2) :]
        try:
            print("模型> ", end="", flush=True)
            reply = (
                client.chat.completions.create(
                    model=model,
                    messages=[system, *recent, user_message],
                    temperature=0.3,
                    max_tokens=max_tokens,
                )
                .choices[0]
                .message.content
                or "(模型没有返回文字)"
            )
            print(reply)
            history = [*recent, user_message, {"role": "assistant", "content": reply}]
        except Exception as exc:
            print(f"\n请求失败：{exc}")
            print("检查服务日志、模型上下文长度和 GPU 显存；可输入 /clear 重试。")
