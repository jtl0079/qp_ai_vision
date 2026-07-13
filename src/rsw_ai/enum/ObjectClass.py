from enum import Enum


class ObjectClass(Enum):
    CAR = (0,)

    @property
    def id(self) -> int:
        return self.value[0]

    @property
    def label(self) -> str:
        return self.name.lower()

    @classmethod
    def from_id(cls, class_id: int):
        for item in cls:
            if item.id == class_id:
                return item
        raise ValueError(f"Unknown class id: {class_id}")

    @classmethod
    def from_label(cls, label: str):
        for item in cls:
            if item.label == label:
                return item
        raise ValueError(f"Unknown label: {label}")
