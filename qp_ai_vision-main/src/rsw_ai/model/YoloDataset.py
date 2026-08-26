from dataclasses import dataclass, field

from rsw_ai.model.YoloDatasetYaml import YoloDatasetYaml


@dataclass
class YoloDataset:
    dataset_txts: dict[str, list[list[float]]] = field(default_factory=dict)
    """
        Dataset_txt structure

        dataset_txt
        └── filename (str)
            └── objects (list)
                └── object_data (list[float])
                    [0] class_id
                    [1] x_center
                    [2] y_center
                    [3] width
                    [4] height

        Example
        -------
        {
            "image1.jpg": [
                [0, 0.52, 0.48, 0.20, 0.15],
                [1, 0.30, 0.65, 0.10, 0.12],
            ],
            "image2.jpg": [
                [2, 0.44, 0.31, 0.18, 0.27],
            ],
        }
    """

    dataset_yml: YoloDatasetYaml = field(default_factory=YoloDatasetYaml)
