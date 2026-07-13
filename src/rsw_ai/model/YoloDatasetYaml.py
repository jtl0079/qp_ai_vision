from dataclasses import dataclass, field
from rsw_ai.enum.ObjectClass import ObjectClass


@dataclass
class YoloDatasetYaml:
    path: str = ""
    train: str = ""
    val: str = ""
    test: str = ""

    names: list[str] = field(default_factory=ObjectClass.all_labels)
