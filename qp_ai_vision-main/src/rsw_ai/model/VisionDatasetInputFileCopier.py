from pathlib import Path

from rsw_ai.interface.Copier import Copier
from rsw_ai.model.VisionDataset import VisionDataset


class VisionDatasetInputFileCopier(Copier[VisionDataset]):
    def copy_dataset(
        self,
        dataset: VisionDataset,
        output_dir: str | Path,
    ) -> None:
        # ====================================
        # Include dependency
        # ====================================

        from rsw_ai.backend.copy_Dataset_input_files import copy_Dataset_input_files

        copy_Dataset_input_files(
            dataset=dataset,
            output_dir=output_dir,
        )
