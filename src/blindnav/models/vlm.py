from __future__ import annotations

from pathlib import Path

from PIL import Image

from blindnav.models.base import LazyModel


class SceneDescriber(LazyModel):
    name = "Qwen3-VL"

    def __init__(self, model_dir: str | Path, *, max_new_tokens: int = 320) -> None:
        super().__init__()
        self.model_dir = Path(model_dir).expanduser().resolve()
        self.max_new_tokens = max_new_tokens

    def _build(self):
        if not (self.model_dir / "config.json").is_file():
            raise FileNotFoundError(f"Qwen model config.json missing in {self.model_dir}")
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        processor = AutoProcessor.from_pretrained(str(self.model_dir), local_files_only=True)
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            str(self.model_dir),
            torch_dtype="auto",
            device_map="auto",
            local_files_only=True,
            attn_implementation="sdpa",
        ).eval()
        return processor, model

    def describe(self, image: Image.Image, prompt: str) -> str:
        import torch

        processor, model = self.load()
        messages = [
            {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}
        ]
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        ).to(model.device)
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        trimmed = generated[:, inputs["input_ids"].shape[1] :]
        return processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()
