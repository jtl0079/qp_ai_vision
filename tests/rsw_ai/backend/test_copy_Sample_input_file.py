

from rsw_ai.backend.copy_Sample_input_file import copy_Sample_input_file
from rsw_ai.model.Sample import Sample


def test_copy_sample_input_file(tmp_path):
    # ---------- Arrange ----------
    src = tmp_path / "input.txt"
    src.write_text("Hello World")

    sample = Sample(
        input=src,
    )

    dst = tmp_path / "output" / "copied.txt"

    # ---------- Act ----------
    copy_Sample_input_file(
        sample=sample,
        output_file_path=dst,
    )

    # ---------- Assert ----------
    assert dst.exists()
    assert dst.read_text() == "Hello World"