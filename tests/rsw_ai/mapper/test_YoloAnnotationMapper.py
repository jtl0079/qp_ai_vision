from rsw_ai.mapper.YoloAnnotationMapper import YoloAnnotationMapper
from rsw_ai.model.BoundingBox import BoundingBox
from rsw_ai.model.DetectionObject import DetectionObject
from rsw_ai.model.YoloAnnotation import YoloAnnotation


def test_from_detection_object():

    detection = DetectionObject(
        class_id=3,
        bbox=BoundingBox(
            x_min=100,
            y_min=50,
            x_max=300,
            y_max=150,
        ),
    )

    annotation = YoloAnnotationMapper.from_DetectionObject(
        detection=detection,
        image_width=1000,
        image_height=500,
    )

    expected = YoloAnnotation(
        class_id=3,
        center_x=(100 + 200 / 2) / 1000,
        center_y=(50 + 100 / 2) / 500,
        width=200 / 1000,
        height=100 / 500,
    )

    assert annotation == expected
