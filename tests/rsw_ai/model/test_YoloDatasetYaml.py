from rsw_ai.enum.ObjectClass import ObjectClass
from rsw_ai.model.YoloDatasetYaml import YoloDatasetYaml


def test_default_names():
    yaml = YoloDatasetYaml()

    assert yaml.names == ObjectClass.labels_sorted_by_id()


def test_to_file_string_all_fields():
    yaml = YoloDatasetYaml(
        path="dataset",
        train="images/train",
        val="images/val",
        test="images/test",
    )

    expected = (
        "path: dataset\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n"
    )

    for i, name in enumerate(ObjectClass.labels_sorted_by_id()):
        expected += f"  {i}: {name}\n"

    assert yaml.to_file_string() == expected


def test_to_file_string_without_optional_fields():
    yaml = YoloDatasetYaml(
        train="images/train",
        val="images/val",
    )

    expected = (
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
    )

    for i, name in enumerate(ObjectClass.labels_sorted_by_id()):
        expected += f"  {i}: {name}\n"

    assert yaml.to_file_string() == expected


def test_custom_names():
    yaml = YoloDatasetYaml(
        train="images/train",
        val="images/val",
        names=["cat", "dog"],
    )

    expected = (
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: cat\n"
        "  1: dog\n"
    )

    assert yaml.to_file_string() == expected