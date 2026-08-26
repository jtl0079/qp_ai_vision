from dataclasses import dataclass, field
from typing import Generic, TypeVar

from rsw_ai.model.Sample import Sample

TInput = TypeVar("TInput")
TTarget = TypeVar("TTarget")


@dataclass
class DatasetSplit(Generic[TInput, TTarget]):
    name: str = ""

    samples: list[Sample[TInput, TTarget]] = field(default_factory=list)
