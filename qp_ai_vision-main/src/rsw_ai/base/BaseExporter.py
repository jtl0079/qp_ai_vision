from abc import ABC

from rsw_ai.interface.Exporter import Exporter, T


class BaseExporter(
    Exporter[T],
    ABC,
):
    def __init__(
        self,
        export_path: str = "",
    ):
        self.export_path = export_path
