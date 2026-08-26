from enum import Enum


class BaseEnum(Enum):
    @property
    def label(self) -> str:
        return self.name.lower()

    @classmethod
    def labels(cls) -> list[str]:
        return [item.label for item in cls]

    @classmethod
    def from_label(cls, label: str):
        for item in cls:
            if item.label == label:
                return item

        raise ValueError(f"{label!r} is not a valid label for {cls.__name__}.")
