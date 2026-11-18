# blindnav — Wearable Navigation Aid for the Blind

AIoT course project. A wearable assistive device that helps blind and visually impaired people walk
outdoors. A body-worn camera sends frames to a GPU server, which:

- **Avoids obstacles**: detects people, vehicles, poles and bikes, and estimates which are nearer
- **Finds the walkable path**: segments sidewalk vs. road
- **Reads signboards**: describes the scene and reads signs aloud (Chinese)
- **Recognizes traffic lights**: locates traffic lights in view (red/green state is in progress)

> ⚠️ **Research prototype.** Depth is *relative* (not meters). Segmentation shows how a region looks,
> not whether it is safe to walk on. This system must not be used alone to decide when to cross a street.

---

## Pipeline

```
 Camera frame (BGR)
     │
     ├─► [1] ObjectDetector   YOLO26            ─► obstacles, traffic lights + left/center/right
     ├─► [2] DepthEstimator   Depth Anything V2 ─► relative depth → which obstacles are nearer
     ├─► [3] SceneSegmenter   YOLO26-sem        ─► sidewalk / road masks + coverage per third
     │
     └─► [4] SceneDescriber   Qwen3-VL-8B       ◄─ image + structured results of [1]–[3]
                    │
                    ▼
        AnalysisResult  →  JSON (server)  /  images + analysis.json (CLI)
```

`Navigator` (in `pipeline.py`) runs these stages. Each model loads lazily the first time it's used.
With `keep_loaded=True` (the server default), models stay in GPU memory between frames.
With `keep_loaded=False` (the CLI default), each model is freed after its stage so the pipeline fits on smaller GPUs.

## Project structure

```
.
├── configs/default.yaml        # model paths and thresholds
├── src/blindnav/
│   ├── cli.py                  # `blindnav` command (analyze / segment / chat / serve)
│   ├── config.py               # Settings dataclasses + YAML/env loading
│   ├── pipeline.py             # Navigator: runs all stages on one frame
│   ├── schemas.py              # Detection, Segmentation, AnalysisResult
│   ├── fusion.py               # depth-to-box fusion, zone coverage
│   ├── prompts.py              # Qwen3-VL prompts (Chinese)
│   ├── visualize.py            # debug drawings (boxes, depth colormap, mask overlay)
│   ├── export.py               # save images + analysis.json
│   ├── chat.py                 # terminal chat with Qwen3-VL via vLLM
│   ├── models/                 # one wrapper per model (lazy loading)
│   │   ├── detector.py         #   YOLO26 detection
│   │   ├── depth.py            #   Depth Anything V2
│   │   ├── segmenter.py        #   YOLO26 semantic segmentation
│   │   └── vlm.py              #   Qwen3-VL (transformers)
│   ├── server/                 # Flask API + web page for the wearable
│   └── utils/                  # geometry, image I/O, device selection
├── tests/                      # pytest suite (runs without GPU or model weights)
├── pyproject.toml
└── requirements.txt
```

## Installation

Use a Linux GPU server (developed on AutoDL; about 20 GB of VRAM is enough for Qwen3-VL-8B) with Python 3.10+.

```bash
# 1. PyTorch matching your CUDA version: https://pytorch.org/get-started/locally/
# 2. This project (editable install, with server, chat and dev extras)
pip install -e ".[all]"
# 3. Optional, only for `blindnav chat`
pip install vllm
```

### Model weights (not stored in git)

Put the weights under one **model root** folder (default `/root/autodl-tmp`):

```
<model_root>/
├── yolo26s.pt                     # downloaded automatically if missing
├── yolo26n-sem.pt
├── Depth-Anything-V2/             # git clone of the repo
│   └── checkpoints/depth_anything_v2_vits.pth
└── Qwen3-VL-8B-Instruct/          # Hugging Face model folder (contains config.json)
```

```bash
cd /root/autodl-tmp
git clone https://github.com/DepthAnything/Depth-Anything-V2
huggingface-cli download Qwen/Qwen3-VL-8B-Instruct --local-dir Qwen3-VL-8B-Instruct
```

To use a different folder, set `export BLINDNAV_MODEL_ROOT=/path`, pass `--model-root /path`,
or copy `configs/default.yaml`, edit it and pass it with `--config`.

## Usage

```bash
blindnav analyze street.jpg                    # all 4 models
blindnav analyze street.jpg --no-vlm           # skip Qwen (fast check)
blindnav analyze street.jpg --no-seg --no-vlm  # detection + depth only
blindnav segment street.jpg                    # sidewalk / road masks only
blindnav chat                                  # Qwen3-VL chat (needs `vllm serve` running)
blindnav serve --port 6006                     # HTTP API for the wearable
```

`blindnav analyze` writes these files to `outputs/<image name>/`:

| File | Content |
|---|---|
| `detected.jpg` | detection boxes |
| `depth.jpg`, `depth.npy` | relative depth (colormap / raw) |
| `sidewalk_mask.png`, `road_mask.png`, `semantic_overlay.jpg` | segmentation |
| `fusion.jpg` | boxes + relative closeness on the segmentation overlay |
| `analysis.json` | all structured results |
| `report.txt` | Qwen3-VL scene description |

### Using it from Python

```python
from blindnav import Navigator, load_settings
from blindnav.utils.image import load_image

nav = Navigator(load_settings("configs/default.yaml"), vlm=True)
result = nav.analyze(load_image("street.jpg"))
print(result.to_dict())
```

### HTTP API

| Method | Path | Description |
|---|---|---|
| GET | `/api/health` | server status |
| POST | `/api/analyze` | fast models only, returns `AnalysisResult` JSON |
| POST | `/api/describe` | same plus a Qwen3-VL description (slow, call it on demand) |

Send the image either as multipart field `image` or as the raw JPEG body:

```bash
curl -F image=@street.jpg http://127.0.0.1:6006/api/analyze
```

On AutoDL, use the custom service (自定义服务) feature on port 6006 to get a public HTTPS URL.

### Interactive chat settings

`blindnav chat` reads these environment variables:

| Variable | Default | Meaning |
|---|---|---|
| `VLM_BASE_URL` | `http://127.0.0.1:8000/v1` | vLLM server address |
| `VLM_MODEL` | first model the server lists | model name |
| `VLM_API_KEY` | `EMPTY` | API key |
| `VLM_MAX_TOKENS` | `512` | longest reply |
| `VLM_MAX_TURNS` | `6` | how many past turns to keep |

## Development

```bash
pytest          # tests use fake models: no GPU or weights needed
ruff check src tests
```

### Migrating from the old scripts

| Old script | New command |
|---|---|
| `navigation4.py --image x` | `blindnav analyze x` |
| `navigation.py --image x` | `blindnav analyze x --no-seg` |
| `fusion.py --image x` | `blindnav analyze x --no-seg --no-vlm` |
| `semantic.py` | `blindnav segment x` |
| `qwen3.py` | `blindnav chat` |

## Roadmap

- [x] Obstacle detection with left / center / right position
- [x] Relative depth fusion
- [x] Sidewalk / road segmentation
- [x] Qwen3-VL scene description and signboard reading
- [x] Unified `Navigator` pipeline, CLI, config file, tests
- [x] HTTP API (`/api/analyze`, `/api/describe`)
- [ ] Traffic-light **state** (red / green) classification
- [ ] Alert rules: turn results into one short spoken sentence
- [ ] Phone web app: camera → API → speech (wearable prototype)
- [ ] Evaluation on our own street photos (accuracy, latency)
- [ ] Hardware distance sensor (ultrasonic / ToF) for obstacle distance in meters
