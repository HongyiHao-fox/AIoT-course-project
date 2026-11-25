"""
语音识别模块：faster-whisper large-v3-turbo
用法：
    python asr.py test.m4a
    python asr.py test.m4a --device cpu
输出：识别结果的 JSON
"""
import argparse
from pathlib import Path

SOURCE_DIR = Path(__file__).parent
import json
import time
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

from faster_whisper import WhisperModel

MODEL_NAME = "large-v3-turbo"


def load_model(device="auto"):
    """优先用 MPS（Apple GPU），失败则退回 CPU int8。"""
    if device in ("auto", "mps"):
        try:
            return WhisperModel(MODEL_NAME, device="mps", compute_type="float16"), "mps"
        except Exception as e:
            if device == "mps":
                raise
            print(f"[asr] MPS 不可用，改用 CPU：{e}")
    if device == "cuda":
        return WhisperModel(MODEL_NAME, device="cuda", compute_type="float16"), "cuda"
    return WhisperModel(MODEL_NAME, device="cpu", compute_type="int8"), "cpu"


def transcribe(model, audio_path, language=None):
    """语音 -> 文字。language=None 表示自动检测（中英混说都可以）。"""
    t0 = time.time()
    segments, info = model.transcribe(
        audio_path,
        beam_size=5,
        language=language,
        vad_filter=True,  # 去掉开头结尾的静音
        initial_prompt="Columbia University, Butler Library",  # 帮助识别地名
    )
    text = "".join(s.text for s in segments).strip()
    return {
        "text": text,
        "language": info.language,
        "language_prob": round(info.language_probability, 3),
        "audio_duration_s": round(info.duration, 2),
        "asr_time_s": round(time.time() - t0, 2),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", nargs="?", default=str(SOURCE_DIR / "test.m4a"), help="音频文件路径（默认 test.m4a）")
    parser.add_argument("--device", default="auto", choices=["auto", "mps", "cpu", "cuda"])
    parser.add_argument("--language", default=None, help="例如 en 或 zh；不填则自动检测")
    args = parser.parse_args()

    t0 = time.time()
    model, used_device = load_model(args.device)
    load_time = round(time.time() - t0, 2)

    result = transcribe(model, args.audio, args.language)
    result.update({"device": used_device, "model_load_time_s": load_time})
    print(json.dumps(result, ensure_ascii=False, indent=2))