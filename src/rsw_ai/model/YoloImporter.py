from rsw_ai.model.YoloDataset import YoloDataset
from rsw_ai.enum.Dataset import Dataset
from rsw_ai.interface.BaseImporter import BaseImporter


class YoloImporter(BaseImporter[YoloDataset]):
    def import_dataset(
        self,
        dataset: Dataset,
        path: str | None = None,
    ) -> YoloDataset:
        # ==========================================
        #           dependancy
        # ==========================================

        from rsw_ai.backend.import_sshikamaru_car_object_detection_csv import (
            import_sshikamaru_car_object_detection_csv,
        )

        # ==========================================
        #           default value
        # ==========================================
        if path is None:
            path = self.import_path

        # ==========================================
        #           logic
        # ==========================================

        dispatch: dict[Dataset, callable] = {
            Dataset.SSHIKAMARU_CAR_OBJECT_DETECTION: import_sshikamaru_car_object_detection_csv,
        }

        try:
            importer = dispatch[dataset]
        except KeyError as exc:
            raise ValueError(f"Unsupported dataset: {dataset}") from exc

        return importer(path)
