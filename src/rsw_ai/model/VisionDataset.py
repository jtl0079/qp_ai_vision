from typing import Generic, TypeVar

from rsw_ai.model.Dataset import Dataset

TTarget = TypeVar("TTarget")


class VisionDataset(Dataset[str, TTarget], Generic[TTarget]):
    pass
