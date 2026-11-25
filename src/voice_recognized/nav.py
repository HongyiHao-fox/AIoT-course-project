import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import requests

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))

DEFAULT_AUDIO = SRC_DIR / "test.m4a"
ORIGIN = (40.8075, -73.9626)  # 起点先写死：116th St & Broadway（Columbia 校门）
SEARCH_RADIUS_M = 2000.0

PLACES_URL = "https://places.googleapis.com/v1/places:searchText"
ROUTES_URL = "https://routes.googleapis.com/directions/v2:computeRoutes"


def get_api_key():
    key = os.environ.get("GOOGLE_MAPS_API_KEY", "").strip()
    if not key:
        sys.exit("缺少 GOOGLE_MAPS_API_KEY，先运行：export GOOGLE_MAPS_API_KEY=你的Key")
    return key


# ---------- 1. 从文字里提取目的地（规则，不用 LLM） ----------
ZH_PREFIX = r"^(请)?(带我去|带我到|我要去|我想去|导航到|导航去|去)\s*"
EN_PREFIX = r"^(please\s+)?(can you\s+)?(take me to|navigate to|go to|guide me to|how do i get to|i want to go to)\s*(the\s+)?"


def extract_destination(text):
    t = text.strip().rstrip("。.!！?？,，")
    t = re.sub(ZH_PREFIX, "", t).strip()
    t = re.sub(EN_PREFIX, "", t, flags=re.I).strip()
    return t or text.strip()


# ---------- 2. Places API (New) Text Search ----------
def search_place(query, api_key):
    resp = requests.post(
        PLACES_URL,
        headers={
            "Content-Type": "application/json",
            "X-Goog-Api-Key": api_key,
            "X-Goog-FieldMask": "places.id,places.displayName,places.formattedAddress,places.location",
        },
        json={
            "textQuery": query,
            "locationBias": {"circle": {
                "center": {"latitude": ORIGIN[0], "longitude": ORIGIN[1]},
                "radius": SEARCH_RADIUS_M,
            }},
        },
        timeout=10,
    )
    resp.raise_for_status()
    places = resp.json().get("places", [])
    if not places:
        return None
    p = places[0]
    return {
        "name": p["displayName"]["text"],
        "address": p.get("formattedAddress", ""),
        "place_id": p.get("id", ""),
        "lat": p["location"]["latitude"],
        "lng": p["location"]["longitude"],
    }


# ---------- 3. Routes API 步行路线 ----------
def walk_route(dest, api_key):
    resp = requests.post(
        ROUTES_URL,
        headers={
            "Content-Type": "application/json",
            "X-Goog-Api-Key": api_key,
            "X-Goog-FieldMask": ",".join([
                "routes.distanceMeters",
                "routes.duration",
                "routes.legs.steps.distanceMeters",
                "routes.legs.steps.staticDuration",
                "routes.legs.steps.navigationInstruction",
                "routes.legs.steps.startLocation",
                "routes.legs.steps.endLocation",
            ]),
        },
        json={
            "origin": {"location": {"latLng": {"latitude": ORIGIN[0], "longitude": ORIGIN[1]}}},
            "destination": {"location": {"latLng": {"latitude": dest["lat"], "longitude": dest["lng"]}}},
            "travelMode": "WALK",
            "languageCode": "en-US",
        },
        timeout=10,
    )
    resp.raise_for_status()
    routes = resp.json().get("routes", [])
    if not routes:
        return None
    route = routes[0]

    steps = []
    for i, s in enumerate(route["legs"][0].get("steps", []), 1):
        nav = s.get("navigationInstruction", {})
        # Google 把附加提示（如 "Take the stairs"）用换行拼在指令后面，这里拆开
        lines = [l for l in nav.get("instructions", "").split("\n") if l]
        steps.append({
            "step": i,
            "maneuver": nav.get("maneuver", ""),
            "instruction": lines[0] if lines else "",
            "notes": lines[1:],
            "distance_m": s.get("distanceMeters", 0),
            "duration": s.get("staticDuration", ""),
            "start": s.get("startLocation", {}).get("latLng", {}),
            "end": s.get("endLocation", {}).get("latLng", {}),
        })
    return {
        "distance_m": route.get("distanceMeters", 0),
        "duration": route.get("duration", ""),
        "steps": steps,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", nargs="?", default=str(DEFAULT_AUDIO), help="音频文件路径")
    parser.add_argument("--text", help="直接输入文字，跳过语音识别")
    parser.add_argument("--device", default="auto", choices=["auto", "mps", "cpu", "cuda"])
    parser.add_argument("-o", "--output", help="把结果另存为 JSON 文件")
    args = parser.parse_args()

    api_key = get_api_key()
    timing = {}

    # 1) 语音 -> 文字
    if args.text:
        asr = {"text": args.text, "source": "text_input"}
    else:
        from asr import load_model, transcribe
        t0 = time.time()
        model, device = load_model(args.device)
        timing["model_load_s"] = round(time.time() - t0, 2)
        asr = transcribe(model, args.audio)
        asr.update({"source": "audio", "audio_file": args.audio, "device": device})
        timing["asr_s"] = asr.pop("asr_time_s")

    # 2) 文字 -> 目的地关键词
    query = extract_destination(asr["text"])

    # 3) Places
    t0 = time.time()
    dest = search_place(query, api_key)
    timing["places_s"] = round(time.time() - t0, 2)

    # 4) Routes
    route = None
    if dest:
        t0 = time.time()
        route = walk_route(dest, api_key)
        timing["routes_s"] = round(time.time() - t0, 2)

    result = {
        "status": "ok" if dest and route else ("place_not_found" if not dest else "route_not_found"),
        "asr": asr,
        "query": query,
        "origin": {"lat": ORIGIN[0], "lng": ORIGIN[1]},
        "destination": dest,
        "route": route,
        "timing": timing,
    }

    out = json.dumps(result, ensure_ascii=False, indent=2)
    print(out)
    if args.output:
        Path(args.output).write_text(out, encoding="utf-8")


if __name__ == "__main__":
    main()