from rsw_ai.mapping.DetectionObject_to_YoloAnnotation import (
    DetectionObject_to_YoloAnnotation,
)
from rsw_ai.model.DatasetSplit import DatasetSplit
from rsw_ai.model.VisionDetectionDataset import VisionDetectionDataset
from rsw_ai.model.VisionYoloDataset import VisionYoloDataset
from rsw_ai.model.Sample import Sample


def VisionDetectionDataset_to_VisionYoloDataset(
    dataset: VisionDetectionDataset,
    image_width: int,
    image_height: int,
) -> VisionYoloDataset:

    if image_width <= 0:
        raise ValueError("image_width must be greater than 0.")

    if image_height <= 0:
        raise ValueError("image_height must be greater than 0.")

    yolo_dataset = VisionYoloDataset(
        name=dataset.name,
        splits=[],
    )

    for split in dataset.splits:

        yolo_samples = []

        for sample in split.samples:

            if sample.target is None:
                raise ValueError("sample.target must not be None.")

            annotations = [
                DetectionObject_to_YoloAnnotation(
                    detection=detection,
                    image_width=image_width,
                    image_height=image_height,
                )
                for detection in sample.target
            ]

            yolo_samples.append(
                Sample(
                    input=sample.input,
                    target=annotations,
                )
            )

        yolo_dataset.splits.append(
            DatasetSplit(
                name=split.name,
                samples=yolo_samples,
            )
        )

    return yolo_dataset