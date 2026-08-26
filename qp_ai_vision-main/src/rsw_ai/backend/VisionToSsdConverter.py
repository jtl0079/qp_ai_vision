from rsw_ai.model.VisionDetectionDataset import VisionDetectionDataset
from rsw_ai.model.SsdDataset import SsdDataset
from rsw_ai.model.SsdAnnotation import SsdAnnotation
from PIL import Image

class VisionToSsdConverter:

    def convert(
        self,
        vision_dataset: VisionDetectionDataset,
    ) -> SsdDataset:

        ssd_dataset = SsdDataset()

        # class names
        ssd_dataset.class_map.names = (
            vision_dataset.class_map.names
        )
        ssd_dataset.name = vision_dataset.name
        ssd_dataset.splits = vision_dataset.splits
        # convert every split
        for split in vision_dataset.splits:

            for sample in split.samples:

                annotation = SsdAnnotation()
                annotation.objects = sample.target

                ssd_dataset.annotations[
                    sample.input
                ] = annotation

                with Image.open(sample.input) as img:
                  ssd_dataset.annotations[sample.input].image_width,ssd_dataset.annotations[sample.input].image_height = img.size

        return ssd_dataset
        