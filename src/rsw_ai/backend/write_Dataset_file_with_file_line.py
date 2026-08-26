from pathlib import Path
from typing import TypeVar

from rsw_ai.interface.SupportsFileLine import SupportsFileLine
from rsw_ai.backend.write_DatasetSplit_file_with_file_line import (
    write_DatasetSplit_file_with_file_line,
)
from rsw_ai.model.Dataset import Dataset

TInput = TypeVar("TInput")
TTarget = TypeVar("TTarget", bound=SupportsFileLine)


def write_Dataset_file_with_file_line(
    dataset: Dataset[TInput, list[TTarget]],
    output_dir: str | Path,
    file_extension: str = ".txt",
) -> None:
    # ====================================
    # Initiate Variable
    # ====================================

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for dataset_split in dataset.splits:
        write_DatasetSplit_file_with_file_line(
            dataset_split=dataset_split,
            output_dir=output_dir / dataset_split.name,
            file_extension=file_extension,
        )
