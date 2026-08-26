from rsw_ai.model.VisionDataset import VisionDataset
from rsw_ai.model.SsdAnnotation import SsdAnnotation


class VisionSsdDataset(VisionDataset[list[SsdAnnotation]]):
    pass
