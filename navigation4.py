#!/usr/bin/env python3
"""Single-image, four-model assistive navigation research demo (not safety-critical navigation)."""
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


def clear_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def mask_overlay(image, class_map, ids):
    out = image.copy()
    for label, color in [('sidewalk', (0, 200, 0)), ('road', (255, 110, 0))]:
        idx = ids.get(label)
        if idx is not None:
            sel = class_map == idx
            if sel.any():
                out[sel] = (0.50 * image[sel] + 0.50 * np.array(color)).astype(np.uint8)
    return out


def run():
    p = argparse.ArgumentParser(description='YOLO26 + Depth Anything V2 + YOLO26-sem + Qwen3-VL')
    p.add_argument('--image', default=str(ROOT / 'test1.jpg'))
    p.add_argument('--output', default=str(ROOT / 'navigation4_output'))
    p.add_argument('--yolo-model', default='yolo26s.pt')
    p.add_argument('--semantic-model', default='yolo26n-sem.pt')
    p.add_argument('--depth-root', default=str(DEPTH_ROOT))
    p.add_argument('--qwen-model', default=str(QWEN_ROOT))
    p.add_argument('--skip-qwen', action='store_true')
    p.add_argument('--conf', type=float, default=0.25)
    p.add_argument('--semantic-imgsz', type=int, default=1024)
    p.add_argument('--max-new-tokens', type=int, default=320)
    args = p.parse_args()

    image_path = Path(args.image).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f'Image not found: {image_path}')
    output.mkdir(parents=True, exist_ok=True)
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise RuntimeError(f'OpenCV cannot decode image: {image_path}')
    H, W = bgr.shape[:2]
    cuda = torch.cuda.is_available()
    device = 'cuda:0' if cuda else 'cpu'
    yolo_device = 0 if cuda else 'cpu'

    from ultralytics import YOLO

    print(f'[1/4] YOLO26 object detection on {device}', flush=True)
    detector = YOLO(args.yolo_model)
    r = detector.predict(str(image_path), device=yolo_device, imgsz=640,
                         conf=args.conf, verbose=False)[0]
    cv2.imwrite(str(output / 'detected.jpg'), r.plot())
    objects = []
    if r.boxes is not None:
        for b in r.boxes:
            x1, y1, x2, y2 = [int(round(float(v))) for v in b.xyxy[0].tolist()]
            x1, x2 = max(0, min(W, x1)), max(0, min(W, x2))
            y1, y2 = max(0, min(H, y1)), max(0, min(H, y2))
            if x2 <= x1 or y2 <= y1:
                continue
            xcenter = (x1+x2)/2/W
            objects.append({
                'class': r.names[int(b.cls[0])],
                'confidence': round(float(b.conf[0]), 3),
                'bbox_xyxy': [x1, y1, x2, y2],
                'image_side': 'left' if xcenter < 0.4 else 'right' if xcenter > 0.6 else 'center',
            })
    del detector, r
    clear_gpu()

    print('[2/4] Depth Anything V2 Small', flush=True)
    depth_root = Path(args.depth_root).expanduser().resolve()
    ckpt = depth_root / 'checkpoints' / 'depth_anything_v2_vits.pth'
    if not ckpt.is_file():
        raise FileNotFoundError(f'Depth checkpoint not found: {ckpt}')
    sys.path.insert(0, str(depth_root))
    from depth_anything_v2.dpt import DepthAnythingV2
    dm = DepthAnythingV2(encoder='vits', features=64, out_channels=[48,96,192,384])
    dm.load_state_dict(torch.load(str(ckpt), map_location='cpu', weights_only=True))
    dm = dm.to(device).eval()
    with torch.inference_mode():
        depth = dm.infer_image(bgr)
    del dm
    clear_gpu()
    dmin, dmax = float(depth.min()), float(depth.max())
    depth_norm = np.uint8(np.clip((depth-dmin)/max(dmax-dmin, 1e-6)*255,0,255))
    cv2.imwrite(str(output/'depth.jpg'), cv2.applyColorMap(depth_norm, cv2.COLORMAP_INFERNO))
    for obj in objects:
        x1,y1,x2,y2 = obj['bbox_xyxy']
        # Inner box patch to reduce contributions from the bounding-box background.
        dx,dy = (x2-x1)//4,(y2-y1)//4
        patch = depth[y1+dy:y2-dy, x1+dx:x2-dx]
        if patch.size:
            obj['relative_depth_score'] = round(float(np.median(patch)),3)

    print('[3/4] YOLO26 semantic segmentation', flush=True)
    segmenter = YOLO(args.semantic_model)
    seg = segmenter.predict(str(image_path), device=yolo_device,
                            imgsz=args.semantic_imgsz, verbose=False)[0]
    if seg.semantic_mask is None:
        raise RuntimeError('semantic_mask is missing: confirm yolo26n-sem.pt and package version')
    cmap = seg.semantic_mask.data.cpu().numpy()
    if cmap.shape != (H,W):
        cmap = cv2.resize(cmap.astype(np.uint8), (W,H), interpolation=cv2.INTER_NEAREST)
    classes = {str(name).strip().lower():int(idx) for idx,name in seg.names.items()}
    cv2.imwrite(str(output/'semantic_classes.png'), cmap.astype(np.uint8))
    sidewalk = np.uint8(cmap == classes['sidewalk'])*255 if 'sidewalk' in classes else np.zeros((H,W),np.uint8)
    road = np.uint8(cmap == classes['road'])*255 if 'road' in classes else np.zeros((H,W),np.uint8)
    cv2.imwrite(str(output/'sidewalk_mask.png'), sidewalk)
    cv2.imwrite(str(output/'road_mask.png'), road)
    overlay = mask_overlay(bgr, cmap, classes)
    cv2.imwrite(str(output/'semantic_overlay.jpg'), overlay)
    fusion = overlay.copy()
    for obj in objects:
        x1,y1,x2,y2=obj['bbox_xyxy']
        cv2.rectangle(fusion,(x1,y1),(x2,y2),(0,0,255),2)
        label = f"{obj['class']} {obj['confidence']:.2f}"
        cv2.putText(fusion,label,(x1,max(18,y1-6)),cv2.FONT_HERSHEY_SIMPLEX,0.55,(0,0,255),2,cv2.LINE_AA)
    cv2.imwrite(str(output/'fusion.jpg'),fusion)

    zone_stats = {}
    for label,mask in [('sidewalk',sidewalk),('road',road)]:
        zone_stats[label+'_image_percent'] = round(float(np.count_nonzero(mask))/mask.size*100,1)
        zone_stats[label+'_left_percent'] = round(float(np.count_nonzero(mask[:,:W//3]))/max(1,mask[:,:W//3].size)*100,1)
        zone_stats[label+'_center_percent'] = round(float(np.count_nonzero(mask[:,W//3:2*W//3]))/max(1,mask[:,W//3:2*W//3].size)*100,1)
        zone_stats[label+'_right_percent'] = round(float(np.count_nonzero(mask[:,2*W//3:]))/max(1,mask[:,2*W//3:].size)*100,1)

    info = {
        'image': str(image_path), 'width': W, 'height': H,
        'models': {'detection':args.yolo_model, 'depth':'Depth Anything V2 Small (relative)',
                   'semantic':args.semantic_model, 'vision_language':str(args.qwen_model)},
        'depth_warning':'Relative depth scores have no meter units and are not calibrated across images.',
        'segmentation_warning':'Sidewalk/road labels describe visual appearance; not safe passage.',
        'zones':zone_stats,
        'objects':objects,
    }
    (output/'analysis.json').write_text(json.dumps(info,indent=2,ensure_ascii=False),encoding='utf-8')
    del segmenter, seg
    clear_gpu()
    print(f'Found {len(objects)} objects; saved detection/depth/semantic/fusion images.',flush=True)

    if args.skip_qwen:
        print('Skipped Qwen. Remove --skip-qwen for all four models.\nOutput: '+str(output))
        return

    print('[4/4] Qwen3-VL-8B local analysis',flush=True)
    qwen_path=Path(args.qwen_model).expanduser().resolve()
    if not (qwen_path/'config.json').is_file():
        raise FileNotFoundError(f'Qwen model config.json missing in {qwen_path}')
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    processor = AutoProcessor.from_pretrained(str(qwen_path), local_files_only=True)
    qwen = Qwen3VLForConditionalGeneration.from_pretrained(
        str(qwen_path), torch_dtype='auto', device_map='auto', local_files_only=True,
        attn_implementation='sdpa').eval()
    obj_data=[{k:v for k,v in item.items() if k in ('class','confidence','image_side','relative_depth_score')}
              for item in objects[:25]]
    prompt=(
        '你是一名研究原型的城市步行场景描述助手，为视障人士提供简洁的环境信息。\n'
        '你会看到原始街景图片和三种模型输出的结构化数据。'
        '请综合图片、目标检测类别位置、相对深度、道路/人行道语义分割占比，'
        '中文回答：(1)可见的障碍物与左右位置；(2)画面中的人行道和道路分布；'
        '(3)无法确定的风险及需要人工确认的事项。最多6句话。\n'
        '严格规则：相对深度分数不代表米数，不跨图片比较；不能从像素占比推断真实可走路径；'
        '禁止声称可以安全前进或允许过街；交通灯检测框不等于行人通行状态；'
        '若模型之间不一致，以不确定方式描述。\n'
        '对象检测：'+json.dumps(obj_data,ensure_ascii=False)+'\n区域统计：'+json.dumps(zone_stats,ensure_ascii=False)
    )
    with Image.open(image_path) as f:
        rgb=f.convert('RGB')
    messages=[{'role':'user','content':[{'type':'image','image':rgb},{'type':'text','text':prompt}]}]
    inputs=processor.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,
                                          return_dict=True,return_tensors='pt').to(qwen.device)
    with torch.inference_mode():
        generation=qwen.generate(**inputs,max_new_tokens=args.max_new_tokens,do_sample=False)
    decoded=processor.batch_decode(generation[:,inputs['input_ids'].shape[1]:],
                                   skip_special_tokens=True,clean_up_tokenization_spaces=False)[0]
    (output/'report.txt').write_text(decoded,encoding='utf-8')
    print('\n=== Qwen report ===\n'+decoded+'\n\nOutput: '+str(output))


if __name__=='__main__':
    run()
