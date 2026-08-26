# tests/rsw_ai/enum/test_Platform.py

from rsw_ai.enum.Platform import Platform


def test_id():
    assert Platform.KAGGLE.id == 0
    assert Platform.CUSTOM.id == 1


def test_label():
    assert Platform.KAGGLE.label == "kaggle"
    assert Platform.CUSTOM.label == "custom"


def test_labels():
    assert Platform.labels() == [
        "kaggle",
        "custom",
    ]