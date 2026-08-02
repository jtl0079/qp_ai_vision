from rsw_ai.model.VisionDataset import VisionDataset
from rsw_ai.model.YoloAnnotation import YoloAnnotation


class VisionYoloDataset(VisionDataset[list[YoloAnnotation]]):
    pass
