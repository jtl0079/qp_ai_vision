from rsw_ai.base.BaseEnum import BaseEnum


class DummyEnum(BaseEnum):
    A = (0,)
    B = (1,)

    @property
    def id(self) -> int:
        return self.value[0]


def test_label():
    assert DummyEnum.A.label == "a"
    assert DummyEnum.B.label == "b"


def test_labels():
    assert DummyEnum.labels() == [
        "a",
        "b",
    ]