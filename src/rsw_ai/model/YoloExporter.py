from pathlib import Path

from rsw_ai.interface.BaseExporter import BaseExporter
from rsw_ai.model.YoloDataset import YoloDataset
from rsw_ai.model.YoloDatasetYaml import YoloDatasetYaml


class YoloExporter(BaseExporter[YoloDataset]):
    def export_dataset(
        self,
        path: str = None,
        dataset: YoloDataset = None,
    ) -> str:

        export_path = Path(path or self.export_path)
        export_path.mkdir(parents=True, exist_ok=True)

        # export all txt labels
        labels_dir = export_path / "labels"
        labels_dir.mkdir(exist_ok=True)

        for filename, dataset_txt in dataset.dataset_txts.items():
            txt_path = labels_dir / f"{Path(filename).stem}.txt"

            self.export_dataset_txt(
                path=str(txt_path),
                dataset_txt=dataset_txt,
            )

        # export dataset.yml
        self.export_dataset_yml(
            path=str(export_path / "dataset.yml"),
            dataset_yml=dataset.dataset_yml,
        )

        return str(export_path)

    def export_dataset_txt(
        self,
        path: str = None,
        dataset_txt: list[list[float]] = None,
    ) -> str:

        output_path = Path(path)

        with output_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            for row in dataset_txt:
                file.write(" ".join(map(str, row)) + "\n")

        return str(output_path)

    def export_dataset_yml(
        self,
        path: str = None,
        dataset_yml: YoloDatasetYaml = None,
    ) -> str:
        """
        Export a YOLO dataset YAML configuration file.

        Parameters
        ----------
        path : str, optional
            Output path of the dataset YAML file.

        dataset_yml : YoloDatasetYaml, optional
            YOLO dataset YAML model containing dataset paths
            and class names.

        Returns
        -------
        str
            Path of the exported YAML file.
        """
        output_path = Path(path)

        with output_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            file.write(f"path: {dataset_yml.path}\n")
            file.write(f"train: {dataset_yml.train}\n")
            file.write(f"val: {dataset_yml.val}\n")

            if dataset_yml.test:
                file.write(f"test: {dataset_yml.test}\n")

            file.write("names:\n")

            for index, name in enumerate(dataset_yml.names):
                file.write(f"  {index}: {name}\n")

        return str(output_path)
