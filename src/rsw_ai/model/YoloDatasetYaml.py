from dataclasses import dataclass, field

from rsw_ai.enum.ObjectClass import ObjectClass
from rsw_ai.interface.SupportsFileString import SupportsFileString


@dataclass
class YoloDatasetYaml(SupportsFileString):
    path: str | None = None
    train: str | None = None
    val: str | None = None
    test: str | None = None

    names: list[str] = field(
        default_factory=ObjectClass.labels_sorted_by_id
    )

    def to_file_string(self) -> str:
        lines: list[str] = []

        if self.path is not None:
            lines.append(f"path: {self.path}")

        if self.train is not None:
            lines.append(f"train: {self.train}")

        if self.val is not None:
            lines.append(f"val: {self.val}")

        if self.test is not None:
            lines.append(f"test: {self.test}")

        lines.append("names:")

        for class_id, class_name in enumerate(self.names):
            lines.append(f"  {class_id}: {class_name}")

        return "\n".join(lines) + "\n"