from pathlib import Path
import csv

import cv2

from rsw_ai.model.YoloDataset import YoloDataset
from rsw_ai.enum.ObjectClass import ObjectClass


def import_sshikamaru_car_object_detection_csv(
    csv_path: str | Path,
) -> YoloDataset:
    csv_path = Path(csv_path)

    image_dir = csv_path.parent / "training_images"

    dataset = YoloDataset()

    with open(csv_path, newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            filename = row["image"]

            image = cv2.imread(str(image_dir / filename))
            image_height, image_width = image.shape[:2]

            xmin = float(row["xmin"])
            ymin = float(row["ymin"])
            xmax = float(row["xmax"])
            ymax = float(row["ymax"])

            width = xmax - xmin
            height = ymax - ymin

            x_center = xmin + width / 2
            y_center = ymin + height / 2

            object_data = [
                ObjectClass.CAR.id,
                x_center / image_width,
                y_center / image_height,
                width / image_width,
                height / image_height,
            ]

            dataset.dataset.setdefault(filename, []).append(object_data)

    return dataset
