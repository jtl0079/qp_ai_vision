from abc import ABC, abstractmethod
from typing import Generic, TypeVar

T = TypeVar("T")


class Exporter(ABC, Generic[T]):
    @abstractmethod
    def export_dataset(
        self,
        path: str,
        dataset: T,
    ) -> str:
        """
        Export dataset to path.

        Returns
        -------
        str
            Exported dataset path.
        """
        pass
