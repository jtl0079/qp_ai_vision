from dataclasses import dataclass, field
from typing import Generic, TypeVar

from rsw_ai.model.DatasetSplit import DatasetSplit

TInput = TypeVar("TInput")
TTarget = TypeVar("TTarget")


@dataclass
class Dataset(Generic[TInput, TTarget]):
    name: str = ""

    splits: list[DatasetSplit[TInput, TTarget]] = field(default_factory=list)
