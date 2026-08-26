from .VisionDetectionDataset import VisionDetectionDataset
from .SsdAnnotation import SsdAnnotation
from dataclasses import dataclass, field

@dataclass
class SsdDataset(VisionDetectionDataset):
    annotations:dict[str, SsdAnnotation] = field(default_factory=dict)
    """
    class_map.names
    dataset.name
    dataset.splits
    annotations
    |_____[string,ssdAnnotation]
    |_____[string,ssdAnnotation]



    ssdAnnotation
    |____image_height(int)
    |____image_width(int)
    |____ DetectionObject(List)
        └── class_id(int)
        └── bbox (list)
                └── [0] x_min
                    [1] y_min
                    [2] x_max
                    [3] y_max
    """
    