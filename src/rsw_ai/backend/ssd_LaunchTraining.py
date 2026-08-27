'''

import sys
sys.path.append("/content/drive/MyDrive/RSW_Y2S1_AI/qp_ai_vision-main/src")

from rsw_ai.model.SsdDataset import SsdDataset
from rsw_ai.model.Dataset import Dataset
from rsw_ai.backend.import_sshikamaru_car_object_detection_dataset import import_sshikamaru_car_object_detection_dataset

from rsw_ai.backend.VisionToSsdConverter import VisionToSsdConverter
from rsw_ai.model.SsdTorchDataset import SsdTorchDataset
import torchvision
from rsw_ai.model.SsdTrainer import SsdTrainer
from rsw_ai.backend.ssdTransform import SsdTransform


data =import_sshikamaru_car_object_detection_dataset("dataSet 的 path")
converter = VisionToSsdConverter()


ssd_data = converter.convert(
    data
)

transfrom = SsdTransform(
    resize=(300, 300),
    horizontal_flip=False,
    vertical_flip=False,
    brightness=1.0,
    contrast=1.0,
    normalize=False
)

train_dataset = SsdTorchDataset(ssd_data,transfrom)

##edges，textures，shapes，higher-level features，shape
model = torchvision.models.detection.ssd300_vgg16(
    num_classes=len(ssd_data.class_map) + 1,
    score_thresh=0.5,
    nms_thresh=0.3,
    detections_per_img=100
)

import torch

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
    epochs=10,
)
'''