from rsw_ai.mapping.DetectionObject_to_YoloAnnotation import (
    DetectionObject_to_YoloAnnotation,
)
from rsw_ai.model.DetectionObject import DetectionObject
from rsw_ai.model.YoloAnnotation import YoloAnnotation


class YoloAnnotationMapper:

    @staticmethod
    def from_DetectionObject(
        detection: DetectionObject,
        image_width: int,
        image_height: int,
    ) -> YoloAnnotation:

        return DetectionObject_to_YoloAnnotation(
            detection=detection,
            image_width=image_width,
            image_height=image_height,
        )