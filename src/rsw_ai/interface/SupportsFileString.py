from abc import ABC, abstractmethod


class SupportsFileString(ABC):

    @abstractmethod
    def to_file_string(self) -> str:
        pass