from dataclasses import dataclass

from rsw_ai.interface.SupportsFileLine import SupportsFileLine


@dataclass
class YoloAnnotation(SupportsFileLine):
    class_id: int = 0

    center_x: float = 0.0
    center_y: float = 0.0

    width: float = 0.0
    height: float = 0.0

    @property
    def is_empty(self) -> bool:
        return self.width == 0 or self.height == 0

    
    # @override
    def to_file_line(self) -> str:
        return (
            f"{self.class_id} "
            f"{self.center_x:.6f} "
            f"{self.center_y:.6f} "
            f"{self.width:.6f} "
            f"{self.height:.6f}"
        )

        
