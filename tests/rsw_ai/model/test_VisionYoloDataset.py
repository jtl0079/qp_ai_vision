from rsw_ai.model.DatasetSplit import DatasetSplit
from rsw_ai.model.VisionYoloDataset import VisionYoloDataset


def test_default_constructor():
    dataset = VisionYoloDataset()

    assert dataset.name == ""
    assert dataset.splits == []


def test_name_constructor():
    dataset = VisionYoloDataset("My YOLO Dataset")

    assert dataset.name == "My YOLO Dataset"
    assert dataset.splits == []


def test_full_constructor():
    train = DatasetSplit(name="train")
    val = DatasetSplit(name="val")

    dataset = VisionYoloDataset(
        name="YOLO Dataset",
        splits=[train, val],
    )

    assert dataset.name == "YOLO Dataset"
    assert len(dataset.splits) == 2
    assert dataset.splits[0] is train
    assert dataset.splits[1] is val


def test_empty_splits():
    dataset = VisionYoloDataset(
        name="Empty Dataset",
        splits=[],
    )

    assert dataset.splits == []


def test_append_split():
    dataset = VisionYoloDataset()

    train = DatasetSplit(name="train")
    dataset.splits.append(train)

    assert len(dataset.splits) == 1
    assert dataset.splits[0].name == "train"