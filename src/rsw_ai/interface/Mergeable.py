from abc import ABC, abstractmethod
from typing import Self


class Mergeable(ABC):
    @abstractmethod
    def merge(
        self,
        other: Self,
    ) -> None:
        """
        Merge another object into this object.

        Parameters
        ----------
        other : Self
            Another object of the same type.
        """
        raise NotImplementedError