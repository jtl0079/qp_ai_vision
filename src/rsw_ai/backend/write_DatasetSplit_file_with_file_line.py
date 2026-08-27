from pathlib import Path

from rsw_ai.model.DatasetSplit import DatasetSplit


def write_DatasetSplit_file_with_file_line(
    dataset_split: DatasetSplit[str, list],
    output_dir: str | Path,
    file_extension: str = ".txt",
) -> None:
    # ====================================
    # Include dependency
    # ====================================
    from rsw_ai.backend.write_sample_file_with_file_line import (
        write_sample_file_with_file_line,
    )

    # ====================================
    # Initiate Variable
    # ====================================

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not file_extension.startswith("."):
        file_extension = f".{file_extension}"

    for sample in dataset_split.samples:
        sample_name = Path(sample.input).stem
        file_path = output_dir / f"{sample_name}{file_extension}"

        write_Sample_file_with_file_line(
            sample,
            file_path,
        )
