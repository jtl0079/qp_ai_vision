import pytest

from rsw_ai.mapping.DetectionObject_to_YoloAnnotation import (
    DetectionObject_to_YoloAnnotation,
)
from rsw_ai.model.BoundingBox import BoundingBox
from rsw_ai.model.DetectionObject import DetectionObject
from rsw_ai.model.YoloAnnotation import YoloAnnotation


def test_convert_detection_object_to_yolo_annotation():

    detection = DetectionObject(
        class_id=3,
        bbox=BoundingBox(
            x_min=100,
            y_min=50,
            x_max=300,
            y_max=150,
        ),
    )

    annotation = DetectionObject_to_YoloAnnotation(
        detection=detection,
        image_width=1000,
        image_height=500,
    )

    expected = YoloAnnotation(
        class_id=3,
        center_x=0.2,
        center_y=0.2,
        width=0.2,
        height=0.2,
    )

    assert annotation == expected


def test_raise_when_image_width_is_zero():

    detection = DetectionObject(
        class_id=0,
        bbox=BoundingBox(),
    )

    with pytest.raises(ValueError, match="image_width must be greater than 0."):
        DetectionObject_to_YoloAnnotation(
            detection=detection,
            image_width=0,
            image_height=100,
        )


def test_raise_when_image_height_is_zero():

    detection = DetectionObject(
        class_id=0,
        bbox=BoundingBox(),
    )

    with pytest.raises(ValueError, match="image_height must be greater than 0."):
        DetectionObject_to_YoloAnnotation(
            detection=detection,
            image_width=100,
            image_height=0,
        )
