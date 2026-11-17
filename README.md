# AIoT Course Project — Wearable Navigation Aid for the Blind

A wearable assistive device that helps blind and visually impaired people walk outdoors.
A body-worn camera captures the street scene, and a vision pipeline:

- **Avoids obstacles** — detects people, vehicles, poles, bikes, etc. and estimates which are closer
- **Finds the walkable path** — segments sidewalk vs. road
- **Reads signboards** — reads street signs and notices aloud (Chinese)
- **Recognizes traffic lights** — locates traffic lights in view (signal-state recognition is in progress)

> ⚠️ **Research prototype.** Depth is *relative* (no meters), segmentation shows appearance, not
> verified safe passage, and the system must not be used alone to decide when to cross a street.

---

## System pipeline

```
 Camera frame
     │
     ├─► [1] YOLO26 detection ───────────► obstacles, traffic lights (+ left/center/right)
     ├─► [2] Depth Anything V2 (Small) ──► relative depth → which obstacles are nearer
     ├─► [3] YOLO26 semantic seg ────────► sidewalk / road masks + left/center/right coverage
     │
     └─► [4] Qwen3-VL-8B-Instruct ◄────── image + structured results from [1]–[3]
                    │
                    ▼
        Short Chinese scene description: obstacles, path, signboard text, uncertain risks
```

## Repository layout

| Path | Purpose |
|---|---|
| `fusion.py` | YOLO26 detection + Depth Anything V2 depth. Outputs each object's position and relative-depth score. |
| `semantic.py` | YOLO26 semantic segmentation: extracts the **sidewalk** mask and a green overlay. |
| `navigation.py` | 3-model pipeline: detection → depth → Qwen3-VL report. |
| `navigation4.py` | **Main demo.** 4-model pipeline: detection → depth → semantic segmentation → Qwen3-VL report. |
| `qwen3.py` | Terminal chat with Qwen3-VL through a running vLLM server; send street images with `/image`. |
| `Server/app.py` | Flask backend for the wearable device to send frames to (skeleton, in progress). |
| `Server/temp/` | Web front-end placeholder (HTML / CSS / JS). |

## Requirements

- Linux GPU server (developed on AutoDL; ~20 GB VRAM recommended for Qwen3-VL-8B)
- Python 3.10+
- Packages:

```bash
pip install torch torchvision ultralytics opencv-python numpy pillow \
            transformers accelerate openai flask
pip install vllm   # only needed for qwen3.py
```

### Model weights (not stored in this repo)

The scripts expect this layout under `/root/autodl-tmp/` (change with command-line flags):

```
/root/autodl-tmp/
├── yolo26s.pt                         # YOLO26 detection
├── yolo26n-sem.pt                     # YOLO26 semantic segmentation
├── Depth-Anything-V2/                 # git clone of the Depth-Anything-V2 repo
│   └── checkpoints/depth_anything_v2_vits.pth
├── Qwen3-VL-8B-Instruct/              # Hugging Face model folder (contains config.json)
└── test.jpg                           # sample street image
```

```bash
cd /root/autodl-tmp
git clone https://github.com/DepthAnything/Depth-Anything-V2
huggingface-cli download Qwen/Qwen3-VL-8B-Instruct --local-dir Qwen3-VL-8B-Instruct
```

## Usage

**Full 4-model demo**

```bash
python navigation4.py --image /root/autodl-tmp/test1.jpg
python navigation4.py --image test1.jpg --skip-qwen      # fast check without Qwen
```

Output in `navigation4_output/`: `detected.jpg`, `depth.jpg`, `semantic_overlay.jpg`, `fusion.jpg`,
`sidewalk_mask.png`, `road_mask.png`, `analysis.json`, `report.txt`.

**Individual stages**

```bash
python fusion.py --image test.jpg            # detection + depth → fusion_output/
python semantic.py                           # sidewalk mask → semantic_output/ (paths set in file)
python navigation.py --image test.jpg        # detection + depth + Qwen → navigation_output/
```

**Interactive Qwen3-VL chat**

```bash
# terminal 1
vllm serve /root/autodl-tmp/Qwen3-VL-8B-Instruct --port 8000
# terminal 2
python qwen3.py
你> /image /root/autodl-tmp/test.jpg
```

Environment variables: `VLM_BASE_URL` (default `http://127.0.0.1:8000/v1`), `VLM_MODEL`,
`VLM_API_KEY`, `VLM_MAX_TOKENS`, `VLM_MAX_TURNS`.

**Web server**

```bash
cd Server && python app.py     # http://127.0.0.1:5000/api
```

## Roadmap

- [x] Obstacle detection with left / center / right position
- [x] Relative depth fusion
- [x] Sidewalk / road segmentation
- [x] Qwen3-VL scene description and signboard reading
- [ ] Traffic-light **state** (red / green) classification
- [ ] `/analyze` API in `Server/` that runs the pipeline on uploaded frames
- [ ] Real-time camera loop (fast models every frame, Qwen on demand)
- [ ] Voice output (TTS) and vibration feedback on the wearable
- [ ] Hardware distance sensor (ultrasonic / ToF) for metric obstacle distance

## Team

AIoT course project.
