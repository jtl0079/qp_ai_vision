"""
src/rsw_ai/training/predict_faster_rcnn.py
-----------------------------------------------
Runs the trained Faster R-CNN on ONE image and saves an annotated copy
with boxes + labels drawn on it. This is the "try it yourself" piece.

Run from the repo root:
    python -m rsw_ai.training.predict_faster_rcnn path/to/photo.jpg

Output is saved next to the input image as "<name>_predicted.jpg".
"""

import sys
import time
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms as T

from rsw_ai.backend.import_nadinpethiyagoda_vehicle_dataset_for_yolo_to_VisionYoloDataset import (
    import_nadinpethiyagoda_vehicle_dataset_for_yolo_to_VisionYoloDataset,
)
from rsw_ai.training.build_faster_rcnn_model import build_faster_rcnn_model
from rsw_ai.training.get_device import get_device
from rsw_ai.training.config import DATASET_ROOT, IMAGE_SIZE, CHECKPOINT_PATH, SCORE_THRESHOLD

BOX_COLORS = ["#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4"]


def _get_class_names() -> list[str]:
    # Reuses the repo's own importer just to read the class_map, so class
    # names/order always match training -- no separate hardcoded list to
    # accidentally get out of sync.
    vision_dataset = import_nadinpethiyagoda_vehicle_dataset_for_yolo_to_VisionYoloDataset(
        dataset_root=DATASET_ROOT,
    )
    return vision_dataset.class_map.names


def predict(image_path: str) -> Path:
    device = get_device()
    class_names = _get_class_names()

    model = build_faster_rcnn_model(len(class_names))
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.to(device).eval()

    image = Image.open(image_path).convert("RGB")
    orig_w, orig_h = image.size

    resized = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    tensor = T.ToTensor()(resized).to(device)

    start = time.time()
    with torch.no_grad():
        output = model([tensor])[0]
    elapsed_ms = (time.time() - start) * 1000

    scale_x, scale_y = orig_w / IMAGE_SIZE, orig_h / IMAGE_SIZE

    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    num_detections = 0
    for box, label, score in zip(output["boxes"], output["labels"], output["scores"]):
        if score.item() < SCORE_THRESHOLD:
            continue
        num_detections += 1

        x1, y1, x2, y2 = box.tolist()
        x1, x2 = x1 * scale_x, x2 * scale_x
        y1, y2 = y1 * scale_y, y2 * scale_y

        class_name = class_names[label.item() - 1]  # -1 undoes the "+1 for background" shift
        color = BOX_COLORS[(label.item() - 1) % len(BOX_COLORS)]

        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        text = f"{class_name} {score.item():.2f}"
        draw.rectangle([x1, max(0, y1 - 14), x1 + 8 * len(text), y1], fill=color)
        draw.text((x1 + 2, max(0, y1 - 14)), text, fill="white", font=font)

        print(f"  {class_name:<12} score={score.item():.2f}  box=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})")

    print(f"[INFO] {num_detections} vehicle(s) found in {elapsed_ms:.1f} ms")

    out_path = Path(image_path).with_name(Path(image_path).stem + "_predicted.jpg")
    annotated.save(out_path)
    print(f"[INFO] Saved annotated image -> {out_path}")
    return out_path


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m rsw_ai.training.predict_faster_rcnn path/to/photo.jpg")
        sys.exit(1)
    predict(sys.argv[1])
