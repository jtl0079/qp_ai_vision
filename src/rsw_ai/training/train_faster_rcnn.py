"""
src/rsw_ai/training/train_faster_rcnn.py
-------------------------------------------
Trains Faster R-CNN on the nadinpethiyagoda vehicle dataset, using this
repo's OWN importer to load the data (no separate loading logic) --
that's the whole point of the data layer this repo already has.

Run from the repo root:
    python -m rsw_ai.training.train_faster_rcnn
(the -m form is needed so the `rsw_ai` package imports resolve correctly)
"""

import time
import torch
from torch.utils.data import DataLoader

from rsw_ai.backend.import_nadinpethiyagoda_vehicle_dataset_for_yolo_to_VisionYoloDataset import (
    import_nadinpethiyagoda_vehicle_dataset_for_yolo_to_VisionYoloDataset,
)
from rsw_ai.training.FasterRCNNDataset import FasterRCNNDataset, collate_fn
from rsw_ai.training.build_faster_rcnn_model import build_faster_rcnn_model
from rsw_ai.training.get_device import get_device
from rsw_ai.training.config import (
    DATASET_ROOT, IMAGE_SIZE, BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE,
    NUM_WORKERS, CHECKPOINT_PATH,
)


def main():
    device = get_device()

    print(f"[INFO] Importing dataset from {DATASET_ROOT} using the repo's own importer...")
    vision_dataset = import_nadinpethiyagoda_vehicle_dataset_for_yolo_to_VisionYoloDataset(
        dataset_root=DATASET_ROOT,
    )
    num_classes = len(vision_dataset.class_map)
    print(f"[INFO] Loaded dataset '{vision_dataset.name}' -- "
          f"{num_classes} classes: {vision_dataset.class_map.names}")

    train_split = next(s for s in vision_dataset.splits if s.name == "train")
    print(f"[INFO] {len(train_split.samples)} training samples")

    train_ds = FasterRCNNDataset(train_split, image_size=IMAGE_SIZE)
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, collate_fn=collate_fn,
    )

    model = build_faster_rcnn_model(num_classes).to(device)
    model.train()

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=LEARNING_RATE, momentum=0.9, weight_decay=5e-4)

    print(f"[INFO] Training Faster R-CNN for {NUM_EPOCHS} epochs on device={device} ...")

    for epoch in range(NUM_EPOCHS):
        start = time.time()
        running_loss = 0.0

        for images, targets in train_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)  # dict of losses in train mode
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            running_loss += losses.item()

        avg_loss = running_loss / max(1, len(train_loader))
        print(f"[Epoch {epoch + 1}/{NUM_EPOCHS}] avg_loss={avg_loss:.4f} "
              f"time={time.time() - start:.1f}s")

    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), CHECKPOINT_PATH)
    print(f"[INFO] Saved trained Faster R-CNN weights -> {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
