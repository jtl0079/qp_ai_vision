from rsw_ai.base.BaseEnum import BaseEnum
from .Platform import Platform


class DatasetRepositoryLayout(BaseEnum):
    # ====================================
    # Enum
    # ====================================

    # id, name, platform
    SSHIKAMARU_CAR_OBJECT_DETECTION = (
        0,
        "sshikamaru/car-object-detection",
        Platform.KAGGLE,
    )

    NADINPETHIYAGODA_VEHICLE_DATASET_FOR_YOLO = (
        1,
        "nadinpethiyagoda/vehicle-dataset-for-yolo",
        Platform.KAGGLE,
    )

    # ====================================
    # Method
    # ====================================

    @property
    def id(self) -> int:
        return self.value[0]

    @property
    def dataset_name(self) -> str:
        return self.value[1]

    @property
    def platform(self) -> Platform:
        raise self.value[2]
