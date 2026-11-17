import argparse
import gc
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path('/root/autodl-tmp')


def position_label(center_x, width):
    ratio = center_x / width
    return 'left' if ratio < 0.4 else 'right' if ratio > 0.6 else 'center'


def core_box(x1, y1, x2, y2, fraction=0.5):
    """Return a centered inner ROI to reduce background contamination."""
    bw, bh = x2 - x1, y2 - y1
    mx, my = int(bw * (1 - fraction) / 2), int(bh * (1 - fraction) / 2)
    return x1 + mx, y1 + my, x2 - mx, y2 - my


def main():
    ap = argparse.ArgumentParser(description='Fuse YOLO detections with Depth Anything V2 relative depth')
    ap.add_argument('--image', default=str(ROOT / 'test.jpg'))
    ap.add_argument('--output', default=str(ROOT / 'fusion_output'))
    ap.add_argument('--yolo-model', default=str(ROOT / 'yolo26s.pt'))
    ap.add_argument('--depth-root', default=str(ROOT / 'Depth-Anything-V2'))
    ap.add_argument('--conf', type=float, default=0.25)
    ap.add_argument('--imgsz', type=int, default=640)
    args = ap.parse_args()

    src = Path(args.image).expanduser().resolve()
    out = Path(args.output).expanduser().resolve()
    if not src.is_file():
        raise FileNotFoundError(f'Image missing: {src}')
    out.mkdir(parents=True, exist_ok=True)
    bgr = cv2.imread(str(src))
    if bgr is None:
        raise ValueError(f'Could not decode: {src}')
    h, w = bgr.shape[:2]
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    print(f'[1/3] Running YOLO on {device}')
    from ultralytics import YOLO
    yolo = YOLO(args.yolo_model)
    result = yolo.predict(source=str(src), device=0 if device.startswith('cuda') else 'cpu',
                          conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
    detections = []
    if result.boxes is not None:
        for box in result.boxes:
            coords = [int(round(v)) for v in box.xyxy[0].tolist()]
            x1, y1, x2, y2 = coords
            x1, y1 = max(0, min(x1, w)), max(0, min(y1, h))
            x2, y2 = max(0, min(x2, w)), max(0, min(y2, h))
            if x2 <= x1 or y2 <= y1:
                continue
            cls = int(box.cls[0].item())
            detections.append({
                'object': str(result.names[cls]),
                'confidence': round(float(box.conf[0].item()), 4),
                'bbox_xyxy': [x1, y1, x2, y2],
                'image_region': position_label((x1 + x2) / 2, w),
            })
    cv2.imwrite(str(out / 'detected.jpg'), result.plot())
    del result, yolo
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print('[2/3] Running Depth Anything V2 Small')
    depth_root = Path(args.depth_root).expanduser().resolve()
    checkpoint = depth_root / 'checkpoints' / 'depth_anything_v2_vits.pth'
    if not checkpoint.is_file():
        raise FileNotFoundError(f'Depth checkpoint missing: {checkpoint}')
    sys.path.insert(0, str(depth_root))
    from depth_anything_v2.dpt import DepthAnythingV2
    depth_model = DepthAnythingV2(encoder='vits', features=64,
                                  out_channels=[48, 96, 192, 384])
    depth_model.load_state_dict(torch.load(str(checkpoint), map_location='cpu', weights_only=True))
    depth_model = depth_model.to(device).eval()
    with torch.inference_mode():
        raw_depth = depth_model.infer_image(bgr)
    raw_depth = np.asarray(raw_depth, dtype=np.float32)
    if raw_depth.shape != (h, w):
        raw_depth = cv2.resize(raw_depth, (w, h), interpolation=cv2.INTER_LINEAR)
    np.save(str(out / 'depth.npy'), raw_depth)
    del depth_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    finite = np.isfinite(raw_depth)
    if not finite.any():
        raise RuntimeError('Depth prediction contains no finite values')
    # Robust image-level normalization for DISPLAY only. Higher means nearer
    # for the relative-depth checkpoint used here, but not an absolute metric.
    p_low, p_high = np.percentile(raw_depth[finite], [5, 95])
    span = max(float(p_high - p_low), 1e-6)
    normalized = np.clip((raw_depth - p_low) / span, 0, 1)
    normalized[~finite] = 0
    colored = cv2.applyColorMap(np.uint8(normalized * 255), cv2.COLORMAP_INFERNO)
    cv2.imwrite(str(out / 'depth.jpg'), colored)

    print('[3/3] Associating detections and depth')
    fusion_vis = bgr.copy()
    for i, det in enumerate(detections, start=1):
        x1, y1, x2, y2 = det['bbox_xyxy']
        a, b, c, d = core_box(x1, y1, x2, y2)
        patch = raw_depth[b:d, a:c]
        valid = patch[np.isfinite(patch)]
        if valid.size:
            score = float(np.median(valid))
            # Normalized per-image relative closeness, for comparing ROIs
            closeness = float(np.clip((score - p_low) / span, 0, 1))
            det['relative_depth_score'] = round(score, 4)
            det['relative_closeness_0_to_1'] = round(closeness, 4)
        else:
            det['relative_depth_score'] = None
            det['relative_closeness_0_to_1'] = None
        det['depth_roi_xyxy'] = [a, b, c, d]
        cv2.rectangle(fusion_vis, (x1, y1), (x2, y2), (0, 230, 50), 2)
        cv2.rectangle(fusion_vis, (a, b), (c, d), (255, 180, 0), 1)
        v = det['relative_closeness_0_to_1']
        label = f"{i}. {det['object']}  rel={v:.2f}" if v is not None else f"{i}. {det['object']}  rel=N/A"
        font_scale = 0.5
        text_y = max(18, y1 - 7)
        cv2.putText(fusion_vis, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(fusion_vis, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.imwrite(str(out / 'fusion_visualization.jpg'), fusion_vis)
    payload = {
        'image': str(src),
        'image_size_wh': [w, h],
        'detection_count': len(detections),
        'depth_note': (
            'Depth Anything V2 Small relative outputs only. Higher raw scores generally '
            'represent nearer content for this checkpoint. relative_closeness_0_to_1 '
            'is normalized within THIS IMAGE and is not a distance, probability, '
            'calibrated hazard threshold, or safe-clearance estimate. '
            'ROI depth may mix object and background pixels.'
        ),
        'objects': detections,
    }
    (out / 'fusion_result.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                                            encoding='utf-8')
    print(f'Fused {len(detections)} objects. Output: {out}')
    print('Files: detected.jpg, depth.jpg, depth.npy, fusion_visualization.jpg, fusion_result.json')


if __name__ == '__main__':
    main()
