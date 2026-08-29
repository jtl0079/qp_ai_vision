import sys
sys.path.insert(0, "/content/qp_ai_vision/src")
import torch
import pickle
from rsw_ai.model.SsdDataset import SsdDataset
from rsw_ai.model.Dataset import Dataset
from rsw_ai.backend.import_nadinpethiyagoda_vehicle_dataset_for_yolo_to_VisionYoloDataset import import_nadinpethiyagoda_vehicle_dataset_for_yolo_to_VisionYoloDataset

from rsw_ai.backend.VisionToSsdConverter import VisionToSsdConverter
from rsw_ai.model.SsdTorchDataset import SsdTorchDataset
import torchvision
from rsw_ai.model.SsdTrainer import SsdTrainer
from rsw_ai.backend.ssdTransform import SsdTransform
from rsw_ai.mapping.VisionYoloDataset_to_VisionDetectionDataset import (
    VisionYoloDataset_to_VisionDetectionDataset
)


def main():
  yolo_data = (
    import_nadinpethiyagoda_vehicle_dataset_for_yolo_to_VisionYoloDataset(
        "/content/drive/MyDrive/RSW_Y2S1_AI/dataset/vehicle dataset"
    )
  )


# ============================================================
# 2. YOLO Dataset -> Detection Dataset
# ============================================================

  detection_data = (
    VisionYoloDataset_to_VisionDetectionDataset(
        yolo_data
    )
  )


# ============================================================
# 3. Detection Dataset -> SSD Dataset
# ============================================================

  converter = VisionToSsdConverter()

  ssd_data = converter.convert(
    detection_data
  )

  with open(
    "/content/drive/MyDrive/RSW_Y2S1_AI/dataset/ssd_data.pkl",
    "wb"
  ) as f:
    pickle.dump(ssd_data, f)
  transfrom = SsdTransform(
    resize=(300, 300),
    horizontal_flip=False,
    vertical_flip=False,
    denoise = "gaussian",
    brightness=1.0,
    contrast=1.0,
    normalize=False
  )

  train_dataset = SsdTorchDataset(ssd_data,transfrom,split="train")

##edges，textures，shapes，higher-level features，shape
  model = torchvision.models.detection.ssd300_vgg16(
    num_classes=len(ssd_data.class_map) + 1,
    score_thresh=0.5,
    nms_thresh=0.3,
    detections_per_img=100
  )

  optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.001,
    momentum=0.9,
    weight_decay=0.0005,
  )

  trainer = SsdTrainer()


  trainer.train(
        train_dataset=train_dataset,
        model=model,
        optimizer=optimizer,
        epochs=50,
  )

if __name__ == "__main__":
    main()
