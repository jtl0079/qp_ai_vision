from pathlib import Path
from typing import Any, TypeVar

from rsw_ai.model.Sample import Sample
from rsw_ai.interface.SupportsFileLine import SupportsFileLine

TTarget = TypeVar("TTarget", bound=SupportsFileLine)


def write_sample_file_with_file_line(
    sample: Sample[Any, TTarget],
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