from abc import abstractmethod



class SupportsFileLine():
    @abstractmethod
    def to_file_line(self) -> str:
        pass

