from rsw_ai.model.VisionDetectionDataset import VisionDetectionDataset
from rsw_ai.model.SsdDataset import SsdDataset
from rsw_ai.model.SsdAnnotation import SsdAnnotation


class VisionToSsdConverter:

    def convert(
        self,
        vision_dataset: VisionDetectionDataset,
    ) -> SsdDataset:

        ssd_dataset = SsdDataset()

        # class names
        ssd_dataset.classes = (
            vision_dataset.class_map.names
        )

        # convert every split
        for split in vision_dataset.splits:

            for sample in split.samples:

                annotation = self.convert_annotation(
                    sample.target
                )

                ssd_dataset.annotations[
                    sample.input
                ] = annotation

        return ssd_dataset


    def convert_annotation(
        self,
        objects,
    ) -> SsdAnnotation:

        annotation = SsdAnnotation()

        for obj in objects:

            bbox = obj.bbox

            # SSD format:
            # [xmin, ymin, xmax, ymax]

            annotation.boxes.append(
                [
                    bbox.x_min,
                    bbox.y_min,
                    bbox.x_max,
                    bbox.y_max,
                ]
            )

            annotation.labels.append(
                obj.class_id
            )

        return annotation