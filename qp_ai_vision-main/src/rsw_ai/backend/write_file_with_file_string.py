from pathlib import Path

from rsw_ai.interface.SupportsFileString import SupportsFileString


def write_file_with_file_string(
    obj: SupportsFileString,
    file_path: str | Path,
) -> None:
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with file_path.open("w", encoding="utf-8") as file:
        file.write(obj.to_file_string())