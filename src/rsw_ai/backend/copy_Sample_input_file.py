from pathlib import Path
import shutil

from rsw_ai.model.Sample import Sample


def copy_Sample_input_file(
    sample: Sample,
    output_file_path: str | Path,
) -> None:

    # ====================================
    # Initiate Variable
    # ====================================

    output_file_path = Path(output_file_path)
    output_file_path.parent.mkdir(parents=True, exist_ok=True)

    shutil.copy2(
        sample.input,
        output_file_path,
    )
