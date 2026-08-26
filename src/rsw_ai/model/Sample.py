from dataclasses import dataclass
from typing import Generic, TypeVar

TInput = TypeVar("TInput")
TTarget = TypeVar("TTarget")


@dataclass
class Sample(Generic[TInput, TTarget]):
    input: TInput | None = None

    target: TTarget | None = None
