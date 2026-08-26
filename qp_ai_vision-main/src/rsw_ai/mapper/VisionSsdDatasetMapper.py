from rsw_ai.mapping.VisionDetectionDataset_to_VisionSsdDataset import (
    VisionDetectionDataset_to_VisionSsdDataset,
)
from rsw_ai.model.VisionDetectionDataset import VisionDetectionDataset
from rsw_ai.model.VisionSsdDataset import VisionSsdDataset


class VisionSsdDatasetMapper:

    @staticmethod
    def from_VisionDetectionDataset(
        dataset: VisionDetectionDataset,
        image_width: int,
        image_height: int,
    ) -> VisionSsdDataset:

        return VisionDetectionDataset_to_VisionSsdDataset(
            dataset=dataset,
            image_width=image_width,
            image_height=image_height,
        )