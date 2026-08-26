from copy import deepcopy

from rsw_ai.mapping.DetectionObject_to_SsdAnnotation import (
    DetectionObject_to_SsdAnnotation,
)
from rsw_ai.model.DatasetSplit import DatasetSplit
from rsw_ai.model.Sample import Sample
from rsw_ai.model.VisionDetectionDataset import VisionDetectionDataset
from rsw_ai.model.VisionSsdDataset import VisionSsdDataset


def VisionDetectionDataset_to_VisionSsdDataset(
    dataset: VisionDetectionDataset,
    image_width: int,
    image_height: int,
) -> VisionSsdDataset:

    if image_width <= 0:
        raise ValueError("image_width must be greater than 0.")

    if image_height <= 0:
        raise ValueError("image_height must be greater than 0.")

    ssd_dataset = VisionSsdDataset(
        name=dataset.name,
        class_map=deepcopy(dataset.class_map),
        splits=[],
    )

    for split in dataset.splits:

        ssd_samples = []

        for sample in split.samples:

            if sample.target is None:
                raise ValueError("sample.target must not be None.")

            annotations = []

            for detection in sample.target:

                ssd_class_id = dataset.class_map.convert_id(
                    class_id=detection.class_id,
                    target=ssd_dataset.class_map,
                )

                annotations.append(
                    DetectionObject_to_SsdAnnotation(
                        detection=detection,
                        class_id=ssd_class_id,
                        image_width=image_width,
                        image_height=image_height,
                    )
                )

            ssd_samples.append(
                Sample(
                    input=sample.input,
                    target=annotations,
                )
            )

        ssd_dataset.splits.append(
            DatasetSplit(
                name=split.name,
                samples=ssd_samples,
            )
        )

    return ssd_dataset