from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from rsw_ai.enum.Dataset import Dataset

T = TypeVar("T")


class Importer(ABC, Generic[T]):
    @abstractmethod
    def import_dataset(
        self,
        path: str,
        dataset: Dataset,
    ) -> T:
        pass
