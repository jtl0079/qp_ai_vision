from enum import Enum


class Platform(Enum):
    KAGGLE = (0,)
    CUSTOM = (1,)

    @property
    def id(self) -> int:
        return int(self.value[0])

    @property
    def label(self) -> str:
        return self.name.lower()
