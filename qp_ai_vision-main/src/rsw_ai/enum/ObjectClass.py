from rsw_ai.base.BaseEnum import BaseEnum


class ObjectClass(BaseEnum):
    CAR = (0,)
    THREEWHEEL = (1,)
    BUS = (2,)
    TRUCK = (3,)
    MOTORBIKE = (4,)
    VAN = (5,)

    @property
    def id(self) -> int:
        return self.value[0]

    @classmethod
    def from_id(cls, class_id: int):
        for item in cls:
            if item.id == class_id:
                return item
        raise ValueError(f"Unknown class id: {class_id}")

    @classmethod
    def labels_sorted_by_id(cls) -> list[str]:
        return [item.label for item in sorted(cls, key=lambda item: item.id)]
