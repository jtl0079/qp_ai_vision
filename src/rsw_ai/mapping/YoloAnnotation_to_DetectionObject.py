
from rsw_ai.model.DetectionObject import DetectionObject
from rsw_ai.model.YoloAnnotation import YoloAnnotation
from rsw_ai.model.BoundingBox import BoundingBox


def YoloAnnotation_to_DetectionObject(
    annotation: YoloAnnotation,
    image_width: int,
    image_height: int,
) -> DetectionObject:

    if image_width <= 0:
        raise ValueError("image_width must be greater than 0.")

    if image_height <= 0:
        raise ValueError("image_height must be greater than 0.")

    # YOLO normalized coordinates -> pixel coordinates

    center_x = annotation.center_x * image_width
    center_y = annotation.center_y * image_height

    width = annotation.width * image_width
    height = annotation.height * image_height

    # Convert center coordinates to corner coordinates

    x_min = center_x - width / 2
    y_min = center_y - height / 2

    x_max = center_x + width / 2
    y_max = center_y + height / 2

    bbox = BoundingBox(
        x_min=x_min,
        y_min=y_min,
        x_max=x_max,
        y_max=y_max,
    )

    return DetectionObject(
        class_id=annotation.class_id,
        bbox=bbox,
    )
