from ultralytics import YOLO
import cv2
import numpy as np
import os

IMAGE = "/root/autodl-tmp/test.jpg"
OUT = "/root/autodl-tmp/semantic_output"
MODEL = "yolo26n-sem.pt"

os.makedirs(OUT, exist_ok=True)

image = cv2.imread(IMAGE)

if image is None:
    raise FileNotFoundError(f"Cannot read image: {IMAGE}")

height, width = image.shape[:2]

print("[1/3] Loading YOLO26 Semantic...")
model = YOLO(MODEL)

print("Supported classes:", model.names)

print("[2/3] Running segmentation...")

result = model.predict(
    source=IMAGE,
    device=0,
    imgsz=1024,
    verbose=False
)[0]

if result.semantic_mask is None:
    raise RuntimeError("No semantic mask returned")

class_map = result.semantic_mask.data.cpu().numpy()

sidewalk_ids = [
    int(idx)
    for idx, name in result.names.items()
    if name.lower() == "sidewalk"
]

if not sidewalk_ids:
    raise RuntimeError(
        f"No sidewalk class. Available: {result.names}"
    )

sidewalk_id = sidewalk_ids[0]

mask = (class_map == sidewalk_id).astype(np.uint8) * 255

mask = cv2.resize(
    mask,
    (width, height),
    interpolation=cv2.INTER_NEAREST
)

overlay = image.copy()

green = np.zeros_like(image)
green[:] = (0, 255, 0)

overlay[mask > 0] = cv2.addWeighted(
    image[mask > 0], 0.5,
    green[mask > 0], 0.5,
    0
)

cv2.imwrite(f"{OUT}/sidewalk_mask.png", mask)
cv2.imwrite(f"{OUT}/sidewalk_overlay.jpg", overlay)

print("[3/3] Finished!")
print("Sidewalk class ID:", sidewalk_id)
print("Mask saved:", f"{OUT}/sidewalk_mask.png")
print("Overlay saved:", f"{OUT}/sidewalk_overlay.jpg")
