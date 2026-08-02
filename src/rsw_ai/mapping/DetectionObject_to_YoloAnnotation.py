from rsw_ai.model.DetectionObject import DetectionObject
from rsw_ai.model.YoloAnnotation import YoloAnnotation


def DetectionObject_to_YoloAnnotation(
    detection: DetectionObject,
    image_width: int,
    image_height: int,
) -> YoloAnnotation:

    if image_width <= 0:
        raise ValueError("image_width must be greater than 0.")

    if image_height <= 0:
        raise ValueError("image_height must be greater than 0.")

    bbox = detection.bbox

    return YoloAnnotation(
        class_id=detection.class_id,
        center_x=bbox.center_x / image_width,
        center_y=bbox.center_y / image_height,
        width=bbox.width / image_width,
        height=bbox.height / image_height,
    )