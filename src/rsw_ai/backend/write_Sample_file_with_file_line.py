from pathlib import Path

from rsw_ai.model.Sample import Sample


def write_Sample_file_with_file_line(
    sample: Sample,
    file_path: str | Path,
) -> None:
    # ====================================
    # Initiate Variable
    # ====================================

    file_path = Path(file_path)

    with file_path.open("w", encoding="utf-8") as file:
        for target in sample.target:
            file.write(target.to_file_line())
            file.write("\n")
