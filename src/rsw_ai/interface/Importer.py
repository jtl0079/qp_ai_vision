from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from rsw_ai.enum.DatasetRepositoryLayout import DatasetRepositoryLayout

T = TypeVar("T")


class Importer(ABC, Generic[T]):
    @abstractmethod
    def import_dataset(
        self,
        path: str,
        dataset_repository_layout: DatasetRepositoryLayout,
    ) -> T:
        pass
