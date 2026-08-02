# test_VisionDetectionDataset.py
from rsw_ai.model.DatasetSplit import DatasetSplit
from rsw_ai.model.VisionDetectionDataset import VisionDetectionDataset


def test_default_constructor():
    dataset = VisionDetectionDataset()

    assert dataset.name == ""
    assert dataset.splits == []


def test_name_constructor():
    dataset = VisionDetectionDataset("My Dataset")

    assert dataset.name == "My Dataset"
    assert dataset.splits == []


def test_full_constructor():
    train = DatasetSplit(name="train")
    val = DatasetSplit(name="val")

    dataset = VisionDetectionDataset(
        name="Car Dataset",
        splits=[train, val],
    )

    assert dataset.name == "Car Dataset"
    assert len(dataset.splits) == 2
    assert dataset.splits[0] is train
    assert dataset.splits[1] is val


def test_empty_splits():
    dataset = VisionDetectionDataset(
        name="Empty Dataset",
        splits=[],
    )

    assert dataset.splits == []


def test_append_split():
    dataset = VisionDetectionDataset()

    train = DatasetSplit(name="train")
    dataset.splits.append(train)

    assert len(dataset.splits) == 1
    assert dataset.splits[0].name == "train"
