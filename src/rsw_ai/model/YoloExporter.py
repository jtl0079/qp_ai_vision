from pathlib import Path

from rsw_ai.backend.write_Dataset_file_with_file_line import (
    write_Dataset_file_with_file_line,
)
from rsw_ai.backend.write_file_with_file_string import (
    write_file_with_file_string,
)
from rsw_ai.interface.Exporter import Exporter
from rsw_ai.mapper.YoloDatasetYamlMapper import (
    YoloDatasetYamlMapper,
)
from rsw_ai.model.VisionDatasetInputFileCopier import (
    VisionDatasetInputFileCopier,
)
from rsw_ai.model.VisionYoloDataset import VisionYoloDataset


class YoloExporter(Exporter[VisionYoloDataset]):
    def export_dataset(
        self,
        path: str | Path,
        dataset: VisionYoloDataset,
    ) -> None:
        # ====================================
        # Initiate Variable
        # ====================================

        path = Path(path)

        # ====================================
        # Logic
        # ====================================

        # Write label files
        write_Dataset_file_with_file_line(
            dataset=dataset,
            output_dir=path / "labels",
        )

        # Copy images
        VisionDatasetInputFileCopier().copy_dataset(
            dataset=dataset,
            output_dir=path / "images",
        )

        # Generate dataset.yaml
        yaml = YoloDatasetYamlMapper.from_VisionYoloDataset(
            dataset=dataset,
        )

        write_file_with_file_string(
            obj=yaml,
            file_path=path / "dataset.yaml",
        )