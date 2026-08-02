from dataclasses import dataclass
from rsw_ai.model.BoundingBox import BoundingBox


@dataclass
class DetectionObject:
    class_id: int 
    bbox: BoundingBox 
