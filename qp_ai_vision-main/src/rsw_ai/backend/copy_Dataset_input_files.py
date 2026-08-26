from pathlib import Path
from typing import TypeVar

from rsw_ai.backend.copy_DatasetSplit_input_files import (
    copy_DatasetSplit_input_files,
)
from rsw_ai.model.Dataset import Dataset

TInput = TypeVar("TInput")
TTarget = TypeVar("TTarget")


def copy_Dataset_input_files(
    dataset: Dataset[TInput, TTarget],
    output_dir: str | Path,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for dataset_split in dataset.splits:
        copy_DatasetSplit_input_files(
            dataset_split=dataset_split,
            output_dir=output_dir / dataset_split.name,
        )