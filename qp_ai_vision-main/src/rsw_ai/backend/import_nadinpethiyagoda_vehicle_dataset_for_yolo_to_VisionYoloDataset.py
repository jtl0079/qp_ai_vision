from pathlib import Path

from rsw_ai.enum.DatasetRepositoryLayout import DatasetRepositoryLayout
from rsw_ai.enum.ObjectClass import ObjectClass
from rsw_ai.model.ClassMap import ClassMap
from rsw_ai.model.DatasetSplit import DatasetSplit
from rsw_ai.model.Sample import Sample
from rsw_ai.model.VisionYoloDataset import VisionYoloDataset
from rsw_ai.model.YoloAnnotation import YoloAnnotation


"""
Dataset structure

vehicle dataset/
├── train/
│   ├── images/
│   └── labels/
└── valid/
    ├── images/
    └── labels/
"""


def import_nadinpethiyagoda_vehicle_dataset_for_yolo_to_VisionYoloDataset(
    dataset_root: str | Path,
) -> VisionYoloDataset:

    dataset_root = Path(dataset_root)

    dataset = VisionYoloDataset(
        name=DatasetRepositoryLayout.NADINPETHIYAGODA_VEHICLE_DATASET_FOR_YOLO.label,
        class_map=ClassMap(
            names=[
                ObjectClass.CAR.label,
                ObjectClass.THREEWHEEL.label,
                ObjectClass.BUS.label,
                ObjectClass.TRUCK.label,
                ObjectClass.MOTORBIKE.label,
                ObjectClass.VAN.label,
            ],
        ),
    )

    for split_name in ("train", "valid"):

        split = DatasetSplit[str, list[YoloAnnotation]](
            name=split_name,
        )

        dataset.splits.append(split)

        image_dir = dataset_root / split_name / "images"
        label_dir = dataset_root / split_name / "labels"

        if not image_dir.is_dir():
            raise NotADirectoryError(image_dir)

        if not label_dir.is_dir():
            raise NotADirectoryError(label_dir)

        for label_path in sorted(label_dir.glob("*.txt")):

            image_path = image_dir / f"{label_path.stem}.jpg"

            if not image_path.exists():
                image_path = image_dir / f"{label_path.stem}.png"

            if not image_path.exists():
                image_path = image_dir / f"{label_path.stem}.jpeg"

            if not image_path.exists():
                raise FileNotFoundError(
                    f"Image not found for {label_path.name}"
                )

            sample = Sample[str, list[YoloAnnotation]](
                input=str(image_path),
                target=[],
            )

            with label_path.open(
                "r",
                encoding="utf-8",
            ) as file:

                for line in file:

                    line = line.strip()

                    if not line:
                        continue

                    class_id, cx, cy, w, h = line.split()

                    sample.target.append(
                        YoloAnnotation(
                            class_id=int(class_id),
                            center_x=float(cx),
                            center_y=float(cy),
                            width=float(w),
                            height=float(h),
                        )
                    )

            split.samples.append(sample)

    return dataset