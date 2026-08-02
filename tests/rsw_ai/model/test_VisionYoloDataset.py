from rsw_ai.model.ClassMap import ClassMap
from rsw_ai.model.DatasetSplit import DatasetSplit
from rsw_ai.model.VisionYoloDataset import VisionYoloDataset


def test_default_constructor():
    dataset = VisionYoloDataset()

    assert dataset.name == ""
    assert dataset.splits == []
    assert dataset.class_map == ClassMap()


def test_name_constructor():
    dataset = VisionYoloDataset("My YOLO Dataset")

    assert dataset.name == "My YOLO Dataset"
    assert dataset.splits == []
    assert dataset.class_map == ClassMap()


def test_full_constructor():
    train = DatasetSplit(name="train")
    val = DatasetSplit(name="val")

    class_map = ClassMap(
        names=["car", "person"],
    )

    dataset = VisionYoloDataset(
        name="YOLO Dataset",
        splits=[train, val],
        class_map=class_map,
    )

    assert dataset.name == "YOLO Dataset"
    assert len(dataset.splits) == 2
    assert dataset.splits[0] is train
    assert dataset.splits[1] is val
    assert dataset.class_map is class_map


def test_empty_splits():
    dataset = VisionYoloDataset(
        name="Empty Dataset",
        splits=[],
    )

    assert dataset.splits == []
    assert dataset.class_map == ClassMap()


def test_append_split():
    dataset = VisionYoloDataset()

    train = DatasetSplit(name="train")
    dataset.splits.append(train)

    assert len(dataset.splits) == 1
    assert dataset.splits[0].name == "train"
    assert dataset.class_map == ClassMap()


def test_constructor_with_class_map():
    class_map = ClassMap(
        names=["car", "person"],
    )

    dataset = VisionYoloDataset(
        class_map=class_map,
    )

    assert dataset.class_map is class_map


def test_default_class_map_is_not_shared():
    dataset1 = VisionYoloDataset()
    dataset2 = VisionYoloDataset()

    dataset1.class_map.names.append("car")

    assert dataset1.class_map.names == ["car"]
    assert dataset2.class_map.names == []