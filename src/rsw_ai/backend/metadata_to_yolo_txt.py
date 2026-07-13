from typing import Any
from rsw_ai.enum.ObjectClass import ObjectClass


def metadata_to_yolo_txt(metadata: dict[str, Any]) -> list[str]:
    """
    Convert metadata dict to YOLO label(.txt) format.

    Parameters
    ----------
    metadata : dict
        Metadata loaded from JSON.

    Returns
    -------
    list[str]
        YOLO label lines.
    """

    image = metadata["image"]
    image_width = image["width"]
    image_height = image["height"]

    yolo_lines = []

    for obj in metadata["objects"]:
        bbox = obj["bbox"]

        xmin = bbox["xmin"]
        ymin = bbox["ymin"]
        xmax = bbox["xmax"]
        ymax = bbox["ymax"]

        bbox_width = xmax - xmin
        bbox_height = ymax - ymin

        x_center = (xmin + xmax) / 2 / image_width
        y_center = (ymin + ymax) / 2 / image_height

        width = bbox_width / image_width
        height = bbox_height / image_height

        class_id = ObjectClass.from_label(obj["class"]).id

        yolo_lines.append(
            f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
        )

    return yolo_lines
