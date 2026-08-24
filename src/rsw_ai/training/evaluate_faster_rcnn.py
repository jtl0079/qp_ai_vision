"""
src/rsw_ai/training/evaluate_faster_rcnn.py
-----------------------------------------------
Runs the trained Faster R-CNN on the "valid" split and computes:
    - Precision, Recall
    - mAP@0.5 (11-point interpolated average precision, averaged over classes)
    - Inference speed (avg ms/image, FPS)

Writes the results to outputs/experiments/faster_rcnn/metrics.json.

Run from the repo root:
    python -m rsw_ai.training.evaluate_faster_rcnn
"""

import json
import time
from collections import defaultdict

import torch
from torch.utils.data import DataLoader

from rsw_ai.backend.import_nadinpethiyagoda_vehicle_dataset_for_yolo_to_VisionYoloDataset import (
    import_nadinpethiyagoda_vehicle_dataset_for_yolo_to_VisionYoloDataset,
)
from rsw_ai.training.FasterRCNNDataset import FasterRCNNDataset, collate_fn
from rsw_ai.training.build_faster_rcnn_model import build_faster_rcnn_model
from rsw_ai.training.get_device import get_device
from rsw_ai.training.config import (
    DATASET_ROOT, IMAGE_SIZE, BATCH_SIZE, NUM_WORKERS,
    CHECKPOINT_PATH, METRICS_PATH, SCORE_THRESHOLD, IOU_THRESHOLD,
)


def box_iou(box1, box2) -> float:
    x1, y1 = max(box1[0], box2[0]), max(box1[1], box2[1])
    x2, y2 = min(box1[2], box2[2]), min(box1[3], box2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def compute_map_precision_recall(all_preds, all_targets, num_classes):
    aps, precisions, recalls = [], [], []

    for cls_id in range(1, num_classes + 1):  # skip background=0
        cls_preds = []
        for img_idx, pred in enumerate(all_preds):
            for box, label, score in zip(pred["boxes"], pred["labels"], pred["scores"]):
                if label.item() == cls_id and score.item() >= SCORE_THRESHOLD:
                    cls_preds.append((img_idx, box.tolist(), score.item()))

        gt_by_image = defaultdict(list)
        total_gt = 0
        for img_idx, tgt in enumerate(all_targets):
            for box, label in zip(tgt["boxes"], tgt["labels"]):
                if label.item() == cls_id:
                    gt_by_image[img_idx].append({"box": box.tolist(), "used": False})
                    total_gt += 1

        if total_gt == 0:
            continue

        cls_preds.sort(key=lambda x: x[2], reverse=True)

        tp, fp = [0] * len(cls_preds), [0] * len(cls_preds)
        for i, (img_idx, box, _score) in enumerate(cls_preds):
            candidates = gt_by_image.get(img_idx, [])
            best_iou, best_j = 0.0, -1
            for j, gt in enumerate(candidates):
                if gt["used"]:
                    continue
                iou = box_iou(box, gt["box"])
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_iou >= IOU_THRESHOLD and best_j >= 0:
                tp[i] = 1
                candidates[best_j]["used"] = True
            else:
                fp[i] = 1

        tp_cum = [sum(tp[: i + 1]) for i in range(len(tp))]
        fp_cum = [sum(fp[: i + 1]) for i in range(len(fp))]
        precision_curve = [t / max(1, (t + f)) for t, f in zip(tp_cum, fp_cum)]
        recall_curve = [t / total_gt for t in tp_cum]

        ap = 0.0
        for t in [i / 10 for i in range(11)]:
            p_at_t = max([p for p, r in zip(precision_curve, recall_curve) if r >= t], default=0.0)
            ap += p_at_t / 11
        aps.append(ap)
        precisions.append(precision_curve[-1] if precision_curve else 0.0)
        recalls.append(recall_curve[-1] if recall_curve else 0.0)

    mAP50 = sum(aps) / len(aps) if aps else 0.0
    precision = sum(precisions) / len(precisions) if precisions else 0.0
    recall = sum(recalls) / len(recalls) if recalls else 0.0
    return precision, recall, mAP50


def main():
    device = get_device()

    vision_dataset = import_nadinpethiyagoda_vehicle_dataset_for_yolo_to_VisionYoloDataset(
        dataset_root=DATASET_ROOT,
    )
    num_classes = len(vision_dataset.class_map)
    valid_split = next(s for s in vision_dataset.splits if s.name == "valid")
    print(f"[INFO] Evaluating on {len(valid_split.samples)} validation samples")

    val_ds = FasterRCNNDataset(valid_split, image_size=IMAGE_SIZE)
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, collate_fn=collate_fn,
    )

    model = build_faster_rcnn_model(num_classes)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.to(device).eval()

    all_preds, all_targets = [], []
    total_time, total_images = 0.0, 0

    with torch.no_grad():
        for images, targets in val_loader:
            images = [img.to(device) for img in images]

            start = time.time()
            outputs = model(images)
            if device.type == "cuda":
                torch.cuda.synchronize()
            total_time += time.time() - start
            total_images += len(images)

            for out, tgt in zip(outputs, targets):
                all_preds.append({k: v.cpu() for k, v in out.items()})
                all_targets.append({k: v.cpu() for k, v in tgt.items()})

    precision, recall, mAP50 = compute_map_precision_recall(all_preds, all_targets, num_classes)
    avg_ms = (total_time / total_images) * 1000
    fps = total_images / total_time if total_time > 0 else 0.0

    result = {
        "algorithm": "Faster R-CNN",
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "mAP50": round(mAP50, 4),
        "avg_inference_time_ms": round(avg_ms, 2),
        "fps": round(fps, 2),
    }

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_PATH, "w") as f:
        json.dump(result, f, indent=2)

    print(result)
    print(f"[INFO] Saved metrics -> {METRICS_PATH}")


if __name__ == "__main__":
    main()
