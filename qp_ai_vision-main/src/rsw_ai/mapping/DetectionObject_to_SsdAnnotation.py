from rsw_ai.model.DetectionObject import DetectionObject
from rsw_ai.model.SsdAnnotation import SsdAnnotation


def DetectionObject_to_SsdAnnotation(
    detection: DetectionObject,
    image_width: int,
    image_height: int,
) -> SsdAnnotation:

    if image_width <= 0:
        raise ValueError("image_width must be greater than 0.")

    if image_height <= 0:
        raise ValueError("image_height must be greater than 0.")

    return SsdAnnotation(
        class_id = detection.class_id,
        bbox = detection.bbox,
        image_height = image_height,
        image_width = image_width
    )