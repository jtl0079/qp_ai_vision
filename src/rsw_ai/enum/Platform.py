from rsw_ai.base.BaseEnum import BaseEnum


class Platform(BaseEnum):
    KAGGLE = (0,)
    CUSTOM = (1,)

    @property
    def id(self) -> int:
        return int(self.value[0])
