from pathlib import Path

from rsw_ai.backend.copy_Sample_input_file import copy_Sample_input_file
from rsw_ai.model.DatasetSplit import DatasetSplit


def copy_DatasetSplit_input_files(
    dataset_split: DatasetSplit[str, list],
    output_dir: str | Path,
) -> None:

    # ====================================
    # Initiate Variable
    # ====================================

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for sample in dataset_split.samples:
        file_name = Path(sample.input).name

        copy_Sample_input_file(
            sample=sample,
            output_file_path=output_dir / file_name,
        )
