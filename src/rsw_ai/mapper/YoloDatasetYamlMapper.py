from rsw_ai.mapping.VisionYoloDataset_to_YoloDatasetYaml import (
    VisionYoloDataset_to_YoloDatasetYaml,
)
from rsw_ai.model.VisionYoloDataset import VisionYoloDataset
from rsw_ai.model.YoloDatasetYaml import YoloDatasetYaml


class YoloDatasetYamlMapper:
    @staticmethod
    def from_VisionYoloDataset(
        dataset: VisionYoloDataset,
    ) -> YoloDatasetYaml:
        return VisionYoloDataset_to_YoloDatasetYaml(dataset)
