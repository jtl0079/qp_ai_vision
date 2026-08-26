from dataclasses import dataclass, field
from .DetectionObject import DetectionObject
@dataclass
class SsdAnnotation:
    objects: list[DetectionObject] = field(default_factory=list)
    image_height:int = 0
    image_width:int = 0

    """
      structure
        DetectionObject(List)
        └── class_id(int)
        └── bbox (list)
                └── [0] x_min
                    [1] y_min
                    [2] x_max
                    [3] y_max
        
    """