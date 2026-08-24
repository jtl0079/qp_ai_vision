"""
src/rsw_ai/training/FasterRCNNDataset.py
------------------------------------------
Bridges this repo's data layer (VisionYoloDataset -> DatasetSplit ->
Sample -> YoloAnnotation, all plain dataclasses) to a real
torch.utils.data.Dataset that torchvision's Faster R-CNN can train on.

Nothing in src/rsw_ai/model/ knows about PyTorch -- that's intentional,
it keeps the data layer reusable for any model later. This file is the
one place that bridges "repo's internal format" -> "what torchvision wants".

torchvision's Faster R-CNN expects, per image:
    boxes:  Tensor[N, 4]  -> [xmin, ymin, xmax, ymax] in PIXEL coordinates
    labels: Tensor[N]     -> class id, where 0 is reserved for "background"
                             (so YoloAnnotation.class_id is shifted +1)
"""

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

from rsw_ai.model.DatasetSplit import DatasetSplit
from rsw_ai.model.YoloAnnotation import YoloAnnotation


class FasterRCNNDataset(Dataset):
    def __init__(
        self,
        split: DatasetSplit[str, list[YoloAnnotation]],
        image_size: int = 320,
    ):
        self.split = split
        self.image_size = image_size
        self.resize = T.Resize((image_size, image_size))
        self.to_tensor = T.ToTensor()

    def __len__(self) -> int:
        return len(self.split.samples)

    def __getitem__(self, idx: int):
        sample = self.split.samples[idx]

        image = Image.open(sample.input).convert("RGB")
        orig_w, orig_h = image.size

        boxes, labels = [], []
        for ann in sample.target:
            # normalized YOLO box -> pixel xyxy box, in the ORIGINAL image size
            xmin = (ann.center_x - ann.width / 2) * orig_w
            ymin = (ann.center_y - ann.height / 2) * orig_h
            xmax = (ann.center_x + ann.width / 2) * orig_w
            ymax = (ann.center_y + ann.height / 2) * orig_h

            # scale to the resized image we're about to produce
            scale_x = self.image_size / orig_w
            scale_y = self.image_size / orig_h
            xmin, xmax = xmin * scale_x, xmax * scale_x
            ymin, ymax = ymin * scale_y, ymax * scale_y

            if xmax <= xmin or ymax <= ymin:
                continue  # skip degenerate boxes

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(ann.class_id + 1)  # +1: 0 is reserved for background

        image = self.resize(image)
        image_tensor = self.to_tensor(image)

        if len(boxes) == 0:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32)
            labels_tensor = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": torch.tensor([idx]),
        }
        return image_tensor, target


def collate_fn(batch):
    """torchvision detection models need a list of images + list of dicts
    (not a stacked tensor), since each image can have a different number
    of boxes."""
    images, targets = zip(*batch)
    return list(images), list(targets)
