from rsw_ai.model.VisionYoloDataset import VisionYoloDataset
from rsw_ai.model.YoloDatasetYaml import YoloDatasetYaml


def VisionYoloDataset_to_YoloDatasetYaml(
    dataset: VisionYoloDataset,
) -> YoloDatasetYaml:

    split_names = {split.name for split in dataset.splits}

    val_dir = None

    if "val" in split_names:
        val_dir = "images/val"

    elif "valid" in split_names:
        val_dir = "images/valid"

    return YoloDatasetYaml(
        train="images/train" if "train" in split_names else None,
        val=val_dir,
        test="images/test" if "test" in split_names else None,
        names=dataset.class_map.names.copy(),
    )