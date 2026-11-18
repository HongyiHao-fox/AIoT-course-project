"""Command-line interface: ``blindnav <command>`` (or ``python -m blindnav``)."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from blindnav.config import Settings, load_settings


def _settings(args: argparse.Namespace) -> Settings:
    settings = load_settings(args.config)
    if args.model_root:
        settings.model_root = args.model_root
    if args.device:
        settings.device = args.device
    return settings


def cmd_analyze(args: argparse.Namespace) -> int:
    from blindnav.export import save_outputs
    from blindnav.pipeline import Navigator
    from blindnav.utils.image import load_image

    settings = _settings(args)
    if args.conf is not None:
        settings.detector.conf = args.conf
    image = load_image(args.image)
    nav = Navigator(settings, segmentation=not args.no_seg, vlm=not args.no_vlm, keep_loaded=args.keep_loaded)
    result = nav.analyze(image)

    out_dir = args.output or Path("outputs") / args.image.stem
    files = save_outputs(result, image, out_dir, source=args.image)
    print(f"Detected {len(result.detections)} objects. Timings (ms): {result.timings_ms}")
    if result.zones:
        print("Zones (% of image):", json.dumps(result.zones, ensure_ascii=False))
    if result.description:
        print("\n=== Scene description ===\n" + result.description + "\n")
    print(f"Saved {len(files)} files to {Path(out_dir).resolve()}")
    return 0


def cmd_segment(args: argparse.Namespace) -> int:
    import cv2

    from blindnav.models import SceneSegmenter
    from blindnav.pipeline import ZONE_CLASSES
    from blindnav.utils.device import resolve_device
    from blindnav.utils.image import load_image
    from blindnav.visualize import overlay_masks

    settings = _settings(args)
    image = load_image(args.image)
    segmenter = SceneSegmenter(
        settings.weights(settings.segmenter.weights),
        device=resolve_device(settings.device),
        imgsz=settings.segmenter.imgsz,
    )
    seg = segmenter.segment(image)
    print("Classes:", seg.class_ids)

    out = args.output or Path("outputs") / f"{args.image.stem}_segment"
    out.mkdir(parents=True, exist_ok=True)
    masks = {}
    for name in ZONE_CLASSES:
        if name not in seg.class_ids:
            print(f"Warning: model has no '{name}' class")
        masks[name] = seg.mask(name)
        cv2.imwrite(str(out / f"{name}_mask.png"), masks[name])
    cv2.imwrite(str(out / "semantic_overlay.jpg"), overlay_masks(image, masks))
    print(f"Saved masks and overlay to {out.resolve()}")
    return 0


def cmd_chat(args: argparse.Namespace) -> int:
    from blindnav.chat import run_chat

    return run_chat(base_url=args.base_url, model=args.model)


def cmd_serve(args: argparse.Namespace) -> int:
    from blindnav.server import create_app

    app = create_app(_settings(args), enable_vlm=not args.no_vlm)
    app.run(host=args.host, port=args.port, threaded=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", type=Path, help="YAML config file (see configs/default.yaml)")
    common.add_argument("--model-root", type=Path, help="Folder holding model weights")
    common.add_argument("--device", help="auto | cpu | cuda:0 | mps")
    common.add_argument("-v", "--verbose", action="store_true", help="Show debug logs")

    parser = argparse.ArgumentParser(prog="blindnav", description="Wearable navigation aid for blind users")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("analyze", parents=[common], help="Run the full pipeline on one image")
    p.add_argument("image", type=Path)
    p.add_argument("-o", "--output", type=Path, help="Output folder (default: outputs/<image name>)")
    p.add_argument("--no-seg", action="store_true", help="Skip sidewalk/road segmentation")
    p.add_argument("--no-vlm", action="store_true", help="Skip the Qwen3-VL description")
    p.add_argument("--conf", type=float, help="Detection confidence threshold")
    p.add_argument("--keep-loaded", action="store_true", help="Keep all models in GPU memory (needs more VRAM)")
    p.set_defaults(func=cmd_analyze)

    p = sub.add_parser("segment", parents=[common], help="Sidewalk/road segmentation only")
    p.add_argument("image", type=Path)
    p.add_argument("-o", "--output", type=Path)
    p.set_defaults(func=cmd_segment)

    p = sub.add_parser("chat", help="Interactive Qwen3-VL chat via a running vLLM server")
    p.add_argument("--base-url", help="Default: $VLM_BASE_URL or http://127.0.0.1:8000/v1")
    p.add_argument("--model", help="Default: first model the server lists")
    p.add_argument("-v", "--verbose", action="store_true")
    p.set_defaults(func=cmd_chat)

    p = sub.add_parser("serve", parents=[common], help="Start the HTTP API for the wearable device")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=6006)
    p.add_argument("--no-vlm", action="store_true", help="Disable /api/describe")
    p.set_defaults(func=cmd_serve)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    return args.func(args)
