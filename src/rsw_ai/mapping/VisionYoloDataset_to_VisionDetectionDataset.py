
from pathlib import Path
from PIL import Image

from rsw_ai.model.VisionYoloDataset import VisionYoloDataset
from rsw_ai.model.VisionDetectionDataset import VisionDetectionDataset
from rsw_ai.model.DatasetSplit import DatasetSplit
from rsw_ai.model.Sample import Sample
from rsw_ai.model.DetectionObject import DetectionObject

from rsw_ai.mapping.YoloAnnotation_to_DetectionObject import (
    YoloAnnotation_to_DetectionObject
)


def VisionYoloDataset_to_VisionDetectionDataset(
    yolo_dataset: VisionYoloDataset,
) -> VisionDetectionDataset:

    detection_dataset = VisionDetectionDataset(
        name=yolo_dataset.name,
        class_map=yolo_dataset.class_map,
    )

    for yolo_split in yolo_dataset.splits:

        detection_split = DatasetSplit[
            str,
            list[DetectionObject]
        ](
            name=yolo_split.name,
        )

        for yolo_sample in yolo_split.samples:

            image_path = Path(yolo_sample.input)

            if not image_path.exists():
                raise FileNotFoundError(image_path)

            with Image.open(image_path) as image:
                image_width, image_height = image.size

            detection_objects = []

            for annotation in yolo_sample.target:

                detection_object = (
                    YoloAnnotation_to_DetectionObject(
                        annotation=annotation,
                        image_width=image_width,
                        image_height=image_height,
                    )
                )

                detection_objects.append(
                    detection_object
                )

            detection_sample = Sample[
                str,
                list[DetectionObject]
            ](
                input=yolo_sample.input,
                target=detection_objects,
            )

            detection_split.samples.append(
                detection_sample
            )

        detection_dataset.splits.append(
            detection_split
        )

    return detection_dataset
