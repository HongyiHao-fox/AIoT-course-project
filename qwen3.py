#!/usr/bin/env python3
"""Qwen3-VL terminal chat for an existing local vLLM OpenAI-compatible server."""
import base64
import io
import os
import sys
from pathlib import Path
from openai import OpenAI

BASE_URL = os.getenv('VLM_BASE_URL', 'http://127.0.0.1:8000/v1')
MODEL = os.getenv('VLM_MODEL', '')
MAX_TOKENS = int(os.getenv('VLM_MAX_TOKENS', '512'))
MAX_TURNS = int(os.getenv('VLM_MAX_TURNS', '6'))

SYSTEM_PROMPT = '''你是一个用于盲人辅助导航原型研究的视觉分析助手。请用简洁中文回答。
当用户提供街景图片时，优先分析：路牌或告示文字、施工围挡、台阶、路缘、障碍物、行人信号等，指出图中相对方位。
清楚区分可直接看见的信息和不确定的推测；看不清文字就说看不清。
不要仅凭图片判断可安全通行、给出精确距离或指示用户过马路；真实避障需要独立测距与安全模块。'''


def image_data_uri(path_str):
    path = Path(os.path.expanduser(path_str)).resolve()
    if not path.is_file():
        raise FileNotFoundError(f'找不到图片：{path}')
    try:
        from PIL import Image, ImageOps
    except ImportError as exc:
        raise RuntimeError('缺少 Pillow，请执行：pip install pillow') from exc
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert('RGB')
        image.thumbnail((1280, 1280))
        buf = io.BytesIO()
        image.save(buf, format='JPEG', quality=85)
    encoded = base64.b64encode(buf.getvalue()).decode('ascii')
    return 'data:image/jpeg;base64,' + encoded


def main():
    client = OpenAI(base_url=BASE_URL, api_key=os.getenv('VLM_API_KEY', 'EMPTY'), timeout=180.0, max_retries=1)
    try:
        models = client.models.list().data
        if not models:
            raise RuntimeError('服务未返回任何模型')
        model = MODEL or models[0].id
    except Exception as exc:
        print(f'无法连接 vLLM 服务：{BASE_URL}\n原因：{exc}\n请先在另一终端启动 vllm serve。')
        sys.exit(1)
    messages = [{'role': 'system', 'content': SYSTEM_PROMPT}]
    print(f'已连接模型：{model}')
    print('直接输入文字聊天；/image 路径 发送图片；/clear 清空历史；/help 帮助；/exit 退出。')
    print('示例：/image /root/autodl-tmp/test.jpg')
    while True:
        try:
            user_input = input('\n你> ').strip()
        except (EOFError, KeyboardInterrupt):
            print('\n已退出。')
            break
        if not user_input:
            continue
        cmd = user_input.lower()
        if cmd in ('/exit', '/quit', 'exit', 'quit'):
            print('已退出。')
            break
        if cmd == '/help':
            print('/image /path/to/photo.jpg 发送图片\n/clear 清空对话\n/exit 退出\n普通文字直接发送')
            continue
        if cmd == '/clear':
            messages = [messages[0]]
            print('对话历史已清空。')
            continue
        if cmd.startswith('/image '):
            path = user_input[7:].strip().strip('"').strip("'")
            try:
                uri = image_data_uri(path)
            except Exception as exc:
                print(f'图片读取失败：{exc}')
                continue
            user_message = {'role': 'user', 'content': [
                {'type': 'text', 'text': '请分析图片中的重要路牌文字、障碍物、相对位置和潜在危险，并明确不确定的信息。'},
                {'type': 'image_url', 'image_url': {'url': uri}},
            ]}
        else:
            user_message = {'role': 'user', 'content': user_input}
        # Keep only recent turns; avoid unbounded history and context overload.
        recent = messages[1:][-(MAX_TURNS * 2):]
        request_messages = [messages[0], *recent, user_message]
        try:
            print('模型> ', end='', flush=True)
            reply = client.chat.completions.create(
                model=model, messages=request_messages,
                temperature=0.3, max_tokens=MAX_TOKENS,
            ).choices[0].message.content or '(模型没有返回文字)'
            print(reply)
            messages = [messages[0], *recent, user_message, {'role': 'assistant', 'content': reply}]
        except Exception as exc:
            print(f'\n请求失败：{exc}')
            print('检查服务日志、模型上下文长度和 GPU 显存；可输入 /clear 重试。')


if __name__ == '__main__':
    main()
