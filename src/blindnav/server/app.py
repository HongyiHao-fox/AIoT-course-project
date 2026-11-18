"""HTTP API the wearable device sends camera frames to.

POST /api/analyze   fast models only (detection, depth, segmentation)
POST /api/describe  same + Qwen3-VL description / sign reading (slow, on demand)

Send the image as multipart field ``image`` or as the raw request body.
"""
from __future__ import annotations

import threading
from collections.abc import Callable
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

from blindnav.config import Settings
from blindnav.pipeline import Navigator
from blindnav.utils.image import decode_image

STATIC_DIR = Path(__file__).parent / "static"


def create_app(
    settings: Settings | None = None,
    *,
    enable_vlm: bool = True,
    navigator_factory: Callable[[], Navigator] | None = None,
) -> Flask:
    app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="/static")
    lock = threading.Lock()  # models are not thread-safe
    state: dict[str, Navigator | None] = {"navigator": None}

    def factory() -> Navigator:
        return Navigator(settings, segmentation=True, vlm=enable_vlm, keep_loaded=True)

    def navigator() -> Navigator:
        if state["navigator"] is None:
            state["navigator"] = (navigator_factory or factory)()
        return state["navigator"]

    def read_image():
        upload = request.files.get("image")
        data = upload.read() if upload else request.get_data()
        if not data:
            return None, (jsonify(error="No image: send multipart field 'image' or a raw JPEG body"), 400)
        try:
            return decode_image(data), None
        except ValueError as exc:
            return None, (jsonify(error=str(exc)), 400)

    def run(describe: bool):
        image, error = read_image()
        if error:
            return error
        with lock:
            nav = navigator()
            if describe and nav.describer is None:
                return jsonify(error="Description is disabled on this server"), 503
            result = nav.analyze(image, describe=describe)
        return jsonify(result.to_dict())

    @app.get("/")
    def index():
        return send_from_directory(STATIC_DIR, "index.html")

    @app.get("/api/health")
    def health():
        return jsonify(status="ok", models_ready=state["navigator"] is not None)

    @app.post("/api/analyze")
    def analyze():
        return run(describe=False)

    @app.post("/api/describe")
    def describe():
        return run(describe=True)

    return app
