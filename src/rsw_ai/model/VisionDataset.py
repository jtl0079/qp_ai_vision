from dataclasses import dataclass, field
from typing import Generic, TypeVar

from rsw_ai.model.ClassMap import ClassMap
from rsw_ai.model.Dataset import Dataset

TTarget = TypeVar("TTarget")


@dataclass
class VisionDataset(
    Dataset[str, TTarget],
    Generic[TTarget],
):
    class_map: ClassMap = field(default_factory=ClassMap)