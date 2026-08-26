from pathlib import Path
import csv

from rsw_ai.enum.DatasetRepositoryLayout import DatasetRepositoryLayout
from rsw_ai.enum.ObjectClass import ObjectClass
from rsw_ai.model.BoundingBox import BoundingBox
from rsw_ai.model.ClassMap import ClassMap
from rsw_ai.model.DatasetSplit import DatasetSplit
from rsw_ai.model.DetectionObject import DetectionObject
from rsw_ai.model.Sample import Sample
from rsw_ai.model.VisionDetectionDataset import VisionDetectionDataset


"""
Original dataset directory structure.

sshikamaru_car_object_detection/
└── data/
    ├── train_solution_bounding_boxes (1).csv
    └── training_images/
        ├── vid_4_780.jpg
        ├── vid_4_800.jpg
        └── ...
"""


def import_sshikamaru_car_object_detection_dataset(
    dataset_root: str | Path,
) -> VisionDetectionDataset:

    # ====================================
    # Initiate Variable
    # ====================================

    dataset_root = Path(dataset_root)

    train_csv = dataset_root / "train_solution_bounding_boxes (1).csv"
    train_image_dir = dataset_root / "training_images"

    if not train_csv.exists():
        raise FileNotFoundError(train_csv)

    if not train_image_dir.is_dir():
        raise NotADirectoryError(train_image_dir)

    dataset = VisionDetectionDataset(
        name=DatasetRepositoryLayout.SSHIKAMARU_CAR_OBJECT_DETECTION.label,
        class_map=ClassMap(
            names=[
                ObjectClass.CAR.label,
            ],
        ),
    )

    train_split = DatasetSplit[str, list[DetectionObject]](
        name="train",
    )

    dataset.splits.append(train_split)

    samples: dict[str, Sample[str, list[DetectionObject]]] = {}

    # ====================================
    # Import CSV
    # ====================================

    with train_csv.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            filename = row["image"]

            sample = samples.get(filename)

            if sample is None:
                sample = Sample(
                    input=str(train_image_dir / filename),
                    target=[],
                )

                samples[filename] = sample
                train_split.samples.append(sample)

            sample.target.append(
                DetectionObject(
                    class_id=dataset.class_map.get_id(
                        ObjectClass.CAR.label,
                    ),
                    bbox=BoundingBox(
                        x_min=float(row["xmin"]),
                        y_min=float(row["ymin"]),
                        x_max=float(row["xmax"]),
                        y_max=float(row["ymax"]),
                    ),
                )
            )

    return dataset