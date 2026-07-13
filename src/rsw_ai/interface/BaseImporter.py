from abc import abstractmethod
from typing import Generic, TypeVar

from rsw_ai.enum.Dataset import Dataset

from .Importer import Importer

T = TypeVar("T")


class BaseImporter(Importer[T], Generic[T]):
    def __init__(self):
        self.import_path: str = ""

    @abstractmethod
    def import_dataset(
        self,
        dataset: Dataset,
        path: str,
    ) -> T:
        pass
