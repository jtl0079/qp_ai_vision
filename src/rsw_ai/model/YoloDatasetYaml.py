from dataclasses import dataclass, field
from rsw_ai.enum.ObjectClass import ObjectClass


@dataclass
class YoloDatasetYaml:
    path: str | None
    train: str | None
    val: str | None
    test: str | None

    names: list[str] = field(default_factory=ObjectClass.labels)

    def to_string(self) -> str:
        lines = []

        if self.path:
            lines.append(f"path: {self.path}")

        if self.train:
            lines.append(f"train: {self.train}")

        if self.val:
            lines.append(f"val: {self.val}")

        if self.test:
            lines.append(f"test: {self.test}")

        lines.append("names:")

        for i, name in enumerate(self.names):
            lines.append(f"  {i}: {name}")

        return "\n".join(lines) + "\n"