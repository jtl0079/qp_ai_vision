from pathlib import Path

from rsw_ai.backend.write_Dataset_file_with_file_line import (
    write_Dataset_file_with_file_line,
)
from rsw_ai.interface.Exporter import Exporter
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
            output_dir=path,
        )

        # Copy images
        VisionDatasetInputFileCopier().copy_dataset(
            dataset=dataset,
            output_dir=path,
        )

        # TODO
        # Generate dataset.yaml
