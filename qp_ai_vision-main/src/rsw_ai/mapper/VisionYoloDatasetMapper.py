from rsw_ai.mapping.VisionDetectionDataset_to_VisionYoloDataset import (
    VisionDetectionDataset_to_VisionYoloDataset,
)
from rsw_ai.model.VisionDetectionDataset import VisionDetectionDataset
from rsw_ai.model.VisionYoloDataset import VisionYoloDataset


class VisionYoloDatasetMapper:

    @staticmethod
    def from_VisionDetectionDataset(
        dataset: VisionDetectionDataset,
        image_width: int,
        image_height: int,
    ) -> VisionYoloDataset:

        return VisionDetectionDataset_to_VisionYoloDataset(
            dataset=dataset,
            image_width=image_width,
            image_height=image_height,
        )