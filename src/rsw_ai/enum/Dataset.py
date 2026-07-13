from enum import Enum

from .Platform import Platform


class Dataset(Enum):
    # id, name, platform
    SSHIKAMARU_CAR_OBJECT_DETECTION = (
        0,
        "sshikamaru/car-object-detection",
        Platform.KAGGLE,
    )

    @property
    def id(self) -> int:
        return self.value[0]

    @property
    def label(self) -> str:
        return self.name.lower()

    @property
    def dataset_name(self) -> str:
        return self.value[1]

    @property
    def platform(self) -> Platform:
        raise self.value[2]
