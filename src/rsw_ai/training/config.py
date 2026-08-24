"""
src/rsw_ai/training/config.py
------------------------------
Settings for the Faster R-CNN training/evaluation/inference pipeline.

Paths follow this repo's own documented layout
(docs/eng_raw/repository_structure_topology.md):

    outputs/
      datasets/downloads/kaggle/...   <- where kagglehub puts the raw dataset
      experiments/faster_rcnn/        <- where THIS pipeline writes checkpoints + metrics
"""

import os
from pathlib import Path

# repo root = four levels up from this file
# (.../src/rsw_ai/training/config.py -> .../qp_ai_vision-main)
PROJECT_ROOT = Path(__file__).resolve().parents[3]

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
EXPERIMENT_DIR = OUTPUTS_DIR / "experiments" / "faster_rcnn"
CHECKPOINT_PATH = EXPERIMENT_DIR / "faster_rcnn.pth"
METRICS_PATH = EXPERIMENT_DIR / "metrics.json"

EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# Dataset location
# ---------------------------------------------------------------------
# Set this env var before running (or hardcode a path here), e.g.:
#   import kagglehub
#   kagglehub_root = kagglehub.dataset_download("nadinpethiyagoda/vehicle-dataset-for-yolo")
# The nadinpethiyagoda dataset nests everything one level down inside a
# folder literally named "vehicle dataset" -- this repo's own importer
# (import_nadinpethiyagoda_vehicle_dataset_for_yolo_to_VisionYoloDataset)
# expects to be pointed AT that folder (the one directly containing
# train/ and valid/), so we auto-detect it the same way as before.
_KAGGLEHUB_ROOT = os.environ.get("VEHICLE_DATASET_ROOT", "")


def _resolve_dataset_root(kagglehub_root: str) -> Path:
    root = Path(kagglehub_root)

    if (root / "train").is_dir():
        return root  # already the right folder

    if root.is_dir():
        for child in root.iterdir():
            if child.is_dir() and (child / "train").is_dir():
                return child

    return root  # importer will raise a clear NotADirectoryError if this is wrong


DATASET_ROOT = _resolve_dataset_root(_KAGGLEHUB_ROOT)

# ---------------------------------------------------------------------
# Training hyperparameters (kept small so a first run finishes quickly --
# raise NUM_EPOCHS once the pipeline is confirmed working end-to-end)
# ---------------------------------------------------------------------
IMAGE_SIZE = 320
BATCH_SIZE = 8
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
NUM_WORKERS = 2
SCORE_THRESHOLD = 0.5   # min confidence to keep a prediction, used at eval/inference time
IOU_THRESHOLD = 0.5     # IoU needed for a prediction to "count" as correct, used in mAP calc
