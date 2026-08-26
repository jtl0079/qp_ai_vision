import os
from typing import Any

import yaml


def create_yolo_yaml_dataset_file(
    file_path: str,
    data: dict[str, Any],
    create_dirs: bool = True,
    encoding: str = "utf-8",
    sort_keys: bool = False,
) -> None:
    """
    Save a dictionary as a YAML file.

    Parameters
    ----------
    file_path : str
        Full path to the output YAML file.

    data : dict[str, Any]
        Dictionary to save.

    create_dirs : bool
        If True, create parent directories if they do not exist.

    encoding : str
        File encoding.

    sort_keys : bool
        Whether to sort dictionary keys.
    """
    if create_dirs:
        os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)

    with open(file_path, "w", encoding=encoding) as f:
        yaml.safe_dump(
            data,
            f,
            default_flow_style=False,
            sort_keys=sort_keys,
            allow_unicode=True,
        )
