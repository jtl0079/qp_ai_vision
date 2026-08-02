from rsw_ai.model.BoundingBox import BoundingBox


def test_width():
    bbox = BoundingBox(10, 20, 30, 60)

    assert bbox.width == 20


def test_height():
    bbox = BoundingBox(10, 20, 30, 60)

    assert bbox.height == 40


def test_center_x():
    bbox = BoundingBox(10, 20, 30, 60)

    assert bbox.center_x == 20


def test_center_y():
    bbox = BoundingBox(10, 20, 30, 60)

    assert bbox.center_y == 40


def test_area():
    bbox = BoundingBox(10, 20, 30, 60)

    assert bbox.area == 800


def test_is_empty_true():
    bbox = BoundingBox()

    assert bbox.is_empty


def test_is_empty_false():
    bbox = BoundingBox(10, 20, 30, 60)

    assert not bbox.is_empty


def test_zero_width():
    bbox = BoundingBox(10, 20, 10, 60)

    assert bbox.width == 0


def test_zero_height():
    bbox = BoundingBox(10, 20, 30, 20)

    assert bbox.height == 0


def test_zero_area():
    bbox = BoundingBox(10, 20, 10, 20)

    assert bbox.area == 0


def test_negative_coordinates():
    bbox = BoundingBox(-5, -10, 5, 10)

    assert bbox.width == 10
    assert bbox.height == 20
    assert bbox.center_x == 0
    assert bbox.center_y == 0
    assert bbox.area == 200