from copy import deepcopy

from rsw_ai.mapping.DetectionObject_to_YoloAnnotation import (
    DetectionObject_to_YoloAnnotation,
)
from rsw_ai.model.DatasetSplit import DatasetSplit
from rsw_ai.model.Sample import Sample
from rsw_ai.model.VisionDetectionDataset import VisionDetectionDataset
from rsw_ai.model.VisionYoloDataset import VisionYoloDataset


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
        class_map=deepcopy(dataset.class_map),
        splits=[],
    )

    for split in dataset.splits:

        yolo_samples = []

        for sample in split.samples:

            if sample.target is None:
                raise ValueError("sample.target must not be None.")

            annotations = []

            for detection in sample.target:

                yolo_class_id = dataset.class_map.convert_id(
                    class_id=detection.class_id,
                    target=yolo_dataset.class_map,
                )

                annotations.append(
                    DetectionObject_to_YoloAnnotation(
                        detection=detection,
                        class_id=yolo_class_id,
                        image_width=image_width,
                        image_height=image_height,
                    )
                )

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