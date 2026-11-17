import argparse
import gc
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

ROOT = Path('/root/autodl-tmp')
DEPTH_ROOT = ROOT / 'Depth-Anything-V2'
QWEN_ROOT = ROOT / 'Qwen3-VL-8B-Instruct'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--image', required=True, help='Path to input jpg/png')
    ap.add_argument('--output', default=str(ROOT / 'navigation_output'))
    ap.add_argument('--yolo-model', default='yolo26s.pt')
    ap.add_argument('--qwen-model', default=str(QWEN_ROOT))
    ap.add_argument('--skip-qwen', action='store_true', help='Test YOLO + depth first')
    args = ap.parse_args()

    src = Path(args.image).expanduser().resolve()
    if not src.is_file():
        raise FileNotFoundError(f'Image not found: {src}')
    out = Path(args.output).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    bgr = cv2.imread(str(src))
    if bgr is None:
        raise ValueError(f'Could not read image: {src}')
    h, w = bgr.shape[:2]
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f'[1/3] YOLO26 on {device}')
    from ultralytics import YOLO
    yolo = YOLO(args.yolo_model)
    result = yolo.predict(source=str(src), device=0 if device.startswith('cuda') else 'cpu', conf=0.25, imgsz=640, verbose=False)[0]
    cv2.imwrite(str(out / 'detected.jpg'), result.plot())
    boxes = []
    if result.boxes is not None:
        for b in result.boxes:
            x1, y1, x2, y2 = [int(round(v)) for v in b.xyxy[0].tolist()]
            x1, x2 = max(0, x1), min(w, x2)
            y1, y2 = max(0, y1), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            center = (x1 + x2) / (2 * w)
            direction = 'left' if center < 0.4 else 'right' if center > 0.6 else 'center'
            boxes.append({
                'object': result.names[int(b.cls[0])],
                'confidence': round(float(b.conf[0]), 3),
                'bbox_xyxy': [x1, y1, x2, y2],
                'image_region': direction,
            })
    del yolo, result
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print('[2/3] Depth Anything V2 Small')
    sys.path.insert(0, str(DEPTH_ROOT))
    from depth_anything_v2.dpt import DepthAnythingV2
    checkpoint = DEPTH_ROOT / 'checkpoints' / 'depth_anything_v2_vits.pth'
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    depth_model = DepthAnythingV2(encoder='vits', features=64, out_channels=[48, 96, 192, 384])
    depth_model.load_state_dict(torch.load(str(checkpoint), map_location='cpu', weights_only=True))
    depth_model = depth_model.to(device).eval()
    with torch.inference_mode():
        depth = depth_model.infer_image(bgr)
    del depth_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    lo, hi = float(np.min(depth)), float(np.max(depth))
    gray = np.uint8(np.clip((depth - lo) / max(hi - lo, 1e-6) * 255, 0, 255))
    colored = cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)
    cv2.imwrite(str(out / 'depth.jpg'), colored)
    for b in boxes:
        x1, y1, x2, y2 = b['bbox_xyxy']
        # Central patch avoids background along large box boundaries.
        px1, px2 = x1 + (x2-x1)//4, x2 - (x2-x1)//4
        py1, py2 = y1 + (y2-y1)//4, y2 - (y2-y1)//4
        patch = depth[py1:py2, px1:px2]
        if patch.size:
            b['relative_depth_score'] = round(float(np.median(patch)), 3)
    payload = {
        'image': str(src),
        'image_size': [w, h],
        'depth_note': 'Raw Depth Anything V2 relative scores. NOT meters; DO NOT infer safe clearance.',
        'objects': boxes,
    }
    (out / 'objects.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Detected {len(boxes)} objects. Saved detected.jpg, depth.jpg, objects.json')

    if args.skip_qwen:
        print('Skipped Qwen. To enable, run without --skip-qwen.')
        return

    print('[3/3] Loading local Qwen3-VL-8B-Instruct')
    qwen_path = Path(args.qwen_model).expanduser().resolve()
    if not (qwen_path / 'config.json').is_file():
        raise FileNotFoundError(f'Qwen model directory has no config.json: {qwen_path}; set --qwen-model to actual model snapshot directory')
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    processor = AutoProcessor.from_pretrained(str(qwen_path), local_files_only=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        str(qwen_path), torch_dtype='auto', device_map='auto', local_files_only=True,
        attn_implementation='sdpa',
    ).eval()
    summary = json.dumps(boxes[:25], ensure_ascii=False)
    prompt = (
        '你是用于研究演示的盲人出行环境描述助手。结合输入图片与以下YOLO检测结果，'
        '简洁地用中文描述：前方可见的主要物体、它们大致位于画面的左中右、需要关注的可见环境特征。'
        '不要断言道路安全、不要给过街许可、不要给出距离米数。'
        '检测结果并非绝对正确；当图片不足以支持判断时，请明确说明不确定。'
        '请输出最多5句话。\nYOLO结果：' + summary
    )
    messages = [{'role': 'user', 'content': [
        {'type': 'image', 'image': Image.open(src).convert('RGB')},
        {'type': 'text', 'text': prompt},
    ]}]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors='pt',
    ).to(model.device)
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=256, do_sample=False)
    trimmed = generated[:, inputs['input_ids'].shape[-1]:]
    text = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    (out / 'report.txt').write_text(text, encoding='utf-8')
    print('\n=== Qwen analysis ===\n' + text)
    print(f'\nAll results: {out}')


if __name__ == '__main__':
    main()
