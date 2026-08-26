from pathlib import Path
from typing import Any

from rsw_ai.enum.ObjectClass import ObjectClass


def create_yolo_dataset_yaml_data(
    dataset_path: str | Path,
    train_path: str = "images/train",
    val_path: str = "images/val",
    test_path: str | None = None,
) -> dict[str, Any]:
    """
    Create a YOLO dataset.yaml configuration.

    Parameters
    ----------
    dataset_path : str | Path
        Root directory of the dataset.

    train_path : str
        Relative path to the training images.

    val_path : str
        Relative path to the validation images.

    test_path : str | None
        Relative path to the test images.

    Returns
    -------
    dict[str, Any]
        YOLO dataset.yaml content.
    """

    yaml_data = {
        "path": str(dataset_path),
        "train": train_path,
        "val": val_path,
    }

    if test_path is not None:
        yaml_data["test"] = test_path

    yaml_data["names"] = {obj.id: obj.label for obj in ObjectClass}

    return yaml_data
