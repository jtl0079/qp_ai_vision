# test_VisionDetectionDataset.py

from rsw_ai.model.ClassMap import ClassMap
from rsw_ai.model.DatasetSplit import DatasetSplit
from rsw_ai.model.VisionDetectionDataset import VisionDetectionDataset


def test_default_constructor():
    dataset = VisionDetectionDataset()

    assert dataset.name == ""
    assert dataset.splits == []
    assert dataset.class_map == ClassMap()


def test_name_constructor():
    dataset = VisionDetectionDataset("My Dataset")

    assert dataset.name == "My Dataset"
    assert dataset.splits == []
    assert dataset.class_map == ClassMap()


def test_full_constructor():
    train = DatasetSplit(name="train")
    val = DatasetSplit(name="val")

    class_map = ClassMap(
        names=["car", "person"],
    )

    dataset = VisionDetectionDataset(
        name="Car Dataset",
        splits=[train, val],
        class_map=class_map,
    )

    assert dataset.name == "Car Dataset"
    assert len(dataset.splits) == 2
    assert dataset.splits[0] is train
    assert dataset.splits[1] is val
    assert dataset.class_map is class_map


def test_empty_splits():
    dataset = VisionDetectionDataset(
        name="Empty Dataset",
        splits=[],
    )

    assert dataset.splits == []
    assert dataset.class_map == ClassMap()


def test_append_split():
    dataset = VisionDetectionDataset()

    train = DatasetSplit(name="train")
    dataset.splits.append(train)

    assert len(dataset.splits) == 1
    assert dataset.splits[0].name == "train"
    assert dataset.class_map == ClassMap()


def test_constructor_with_class_map():
    class_map = ClassMap(
        names=["car", "person"],
    )

    dataset = VisionDetectionDataset(
        class_map=class_map,
    )

    assert dataset.class_map is class_map


def test_default_class_map_is_not_shared():
    dataset1 = VisionDetectionDataset()
    dataset2 = VisionDetectionDataset()

    dataset1.class_map.names.append("car")

    assert dataset1.class_map.names == ["car"]
    assert dataset2.class_map.names == []