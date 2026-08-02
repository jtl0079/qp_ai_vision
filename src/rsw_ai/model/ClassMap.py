from dataclasses import dataclass, field


@dataclass
class ClassMap:
    names: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.names)

    def add(self, class_name: str) -> int:
        if class_name in self.names:
            return self.get_id(class_name)

        self.names.append(class_name)
        return len(self.names) - 1

    def get_name(self, class_id: int) -> str:
        return self.names[class_id]

    def get_id(self, class_name: str) -> int:
        return self.names.index(class_name)

    def contains(self, class_name: str) -> bool:
        return class_name in self.names

    def validate(self) -> None:
        if len(set(self.names)) != len(self.names):
            raise ValueError("Duplicate class names found.")

    def convert_id(
        self,
        class_id: int,
        target: "ClassMap",
    ) -> int:
        class_name = self.get_name(class_id)
        if not target.contains(class_name):
            raise ValueError(f'Target ClassMap does not contain class "{class_name}".')
        return target.get_id(class_name)
