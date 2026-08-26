from rsw_ai.mapping.DetectionObject_to_SsdAnnotation import (
    DetectionObject_to_SsdAnnotation,
)
from rsw_ai.model.DetectionObject import DetectionObject
from rsw_ai.model.SsdAnnotation import SsdAnnotation


class SsdAnnotationMapper:
    @staticmethod
    def from_DetectionObject(
        detection: DetectionObject,
        image_width: int,
        image_height: int,
    ) -> SsdAnnotation:
        return DetectionObject_to_SsdAnnotation(
            detection=detection,
            image_width=image_width,
            image_height=image_height,
        )
