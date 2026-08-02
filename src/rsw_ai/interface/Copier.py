from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generic, TypeVar

TDataset = TypeVar("TDataset")


class Copier(ABC, Generic[TDataset]):
    @abstractmethod
    def copy_dataset(
        self,
        dataset: TDataset,
        output_dir: str | Path,
    ) -> None:
        pass
