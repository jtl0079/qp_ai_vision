"""
src/rsw_ai/training/build_faster_rcnn_model.py
-------------------------------------------------
Builds a Faster R-CNN model via transfer learning: start from
COCO-pretrained weights, replace the final classification head with one
sized for this project's classes. This is why a good model can come out
of only ~2100 training images instead of needing millions.
"""

from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def build_faster_rcnn_model(num_classes: int):
    """
    num_classes: number of REAL classes (e.g. 6 for car/threewheel/bus/
    truck/motorbike/van) -- background is added automatically, matching
    the +1 shift used in FasterRCNNDataset.
    """
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)
    return model
