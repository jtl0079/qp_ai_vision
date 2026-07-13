from pathlib import Path


def write_text_file(
    output_file: str | Path,
    lines: list[str],
) -> None:
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )
