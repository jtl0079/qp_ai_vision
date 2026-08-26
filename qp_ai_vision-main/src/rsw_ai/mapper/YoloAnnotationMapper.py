from rsw_ai.mapping.DetectionObject_to_YoloAnnotation import (
    DetectionObject_to_YoloAnnotation,
)
from rsw_ai.model.DetectionObject import DetectionObject
from rsw_ai.model.YoloAnnotation import YoloAnnotation


class YoloAnnotationMapper:
    @staticmethod
    def from_DetectionObject(
        detection: DetectionObject,
        class_id: int,
        image_width: int,
        image_height: int,
    ) -> YoloAnnotation:
        return DetectionObject_to_YoloAnnotation(
            detection=detection,
            class_id=class_id,
            image_width=image_width,
            image_height=image_height,
        )
