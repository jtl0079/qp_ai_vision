# test_YoloAnnotation.py

from rsw_ai.model.YoloAnnotation import YoloAnnotation


# ====================================
# Constructor
# ====================================


def test_default_constructor():
    annotation = YoloAnnotation()

    assert annotation.class_id == 0
    assert annotation.center_x == 0.0
    assert annotation.center_y == 0.0
    assert annotation.width == 0.0
    assert annotation.height == 0.0


def test_full_constructor():
    annotation = YoloAnnotation(
        class_id=1,
        center_x=0.5,
        center_y=0.6,
        width=0.2,
        height=0.3,
    )

    assert annotation.class_id == 1
    assert annotation.center_x == 0.5
    assert annotation.center_y == 0.6
    assert annotation.width == 0.2
    assert annotation.height == 0.3


# ====================================
# Property
# ====================================


def test_is_empty_true():
    annotation = YoloAnnotation()

    assert annotation.is_empty


def test_is_empty_false():
    annotation = YoloAnnotation(
        class_id=1,
        center_x=0.5,
        center_y=0.5,
        width=0.4,
        height=0.2,
    )

    assert not annotation.is_empty


def test_is_empty_when_width_is_zero():
    annotation = YoloAnnotation(
        class_id=1,
        center_x=0.5,
        center_y=0.5,
        width=0.0,
        height=0.2,
    )

    assert annotation.is_empty


def test_is_empty_when_height_is_zero():
    annotation = YoloAnnotation(
        class_id=1,
        center_x=0.5,
        center_y=0.5,
        width=0.2,
        height=0.0,
    )

    assert annotation.is_empty


# ====================================
# interfcae File Line Formatting
# ====================================


def test_to_file_line():
    annotation = YoloAnnotation(
        class_id=1,
        center_x=0.5,
        center_y=0.6,
        width=0.2,
        height=0.3,
    )

    assert (
        annotation.to_file_line()
        == "1 0.500000 0.600000 0.200000 0.300000"
    )


def test_to_file_line_default():
    annotation = YoloAnnotation()

    assert (
        annotation.to_file_line()
        == "0 0.000000 0.000000 0.000000 0.000000"
    )