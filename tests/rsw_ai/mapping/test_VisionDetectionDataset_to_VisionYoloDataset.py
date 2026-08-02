import pytest

from rsw_ai.mapping.VisionDetectionDataset_to_VisionYoloDataset import (
    VisionDetectionDataset_to_VisionYoloDataset,
)
from rsw_ai.model.BoundingBox import BoundingBox
from rsw_ai.model.DatasetSplit import DatasetSplit
from rsw_ai.model.DetectionObject import DetectionObject
from rsw_ai.model.Sample import Sample
from rsw_ai.model.VisionDetectionDataset import VisionDetectionDataset
from rsw_ai.model.YoloAnnotation import YoloAnnotation


def test_VisionDetectionDataset_to_VisionYoloDataset():

    dataset = VisionDetectionDataset(
        name="test_dataset",
        splits=[
            DatasetSplit(
                name="train",
                samples=[
                    Sample(
                        input="train_001.jpg",
                        target=[
                            DetectionObject(
                                class_id=0,
                                bbox=BoundingBox(
                                    x_min=100,
                                    y_min=50,
                                    x_max=300,
                                    y_max=150,
                                ),
                            ),
                        ],
                    ),
                    Sample(
                        input="train_002.jpg",
                        target=[
                            DetectionObject(
                                class_id=1,
                                bbox=BoundingBox(
                                    x_min=200,
                                    y_min=100,
                                    x_max=600,
                                    y_max=300,
                                ),
                                ),
                            DetectionObject(
                                class_id=2,
                                bbox=BoundingBox(
                                    x_min=400,
                                    y_min=200,
                                    x_max=800,
                                    y_max=400,
                                ),
                            ),
                        ],
                    ),
                ],
            ),
            DatasetSplit(
                name="val",
                samples=[
                    Sample(
                        input="val_001.jpg",
                        target=[
                            DetectionObject(
                                class_id=0,
                                bbox=BoundingBox(
                                    x_min=0,
                                    y_min=0,
                                    x_max=500,
                                    y_max=250,
                                ),
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )

    result = VisionDetectionDataset_to_VisionYoloDataset(
        dataset=dataset,
        image_width=1000,
        image_height=500,
    )

    # Dataset name
    assert result.name == "test_dataset"

    # Number of splits
    assert len(result.splits) == 2

    # Split names
    assert result.splits[0].name == "train"
    assert result.splits[1].name == "val"

    # -------------------------
    # train split
    # -------------------------

    train_split = result.splits[0]

    assert len(train_split.samples) == 2

    # train_001.jpg
    train_sample_1 = train_split.samples[0]

    assert train_sample_1.input == "train_001.jpg"

    assert train_sample_1.target == [
        YoloAnnotation(
            class_id=0,
            center_x=0.2,
            center_y=0.2,
            width=0.2,
            height=0.2,
        )
    ]

    # train_002.jpg
    train_sample_2 = train_split.samples[1]

    assert train_sample_2.input == "train_002.jpg"

    assert train_sample_2.target == [
        YoloAnnotation(
            class_id=1,
            center_x=0.4,
            center_y=0.4,
            width=0.4,
            height=0.4,
        ),
        YoloAnnotation(
            class_id=2,
            center_x=0.6,
            center_y=0.6,
            width=0.4,
            height=0.4,
        ),
    ]

    # -------------------------
    # val split
    # -------------------------

    val_split = result.splits[1]

    assert len(val_split.samples) == 1

    val_sample = val_split.samples[0]

    assert val_sample.input == "val_001.jpg"

    assert val_sample.target == [
        YoloAnnotation(
            class_id=0,
            center_x=0.25,
            center_y=0.25,
            width=0.5,
            height=0.5,
        )
    ]


def test_raise_when_image_width_is_zero():

    dataset = VisionDetectionDataset(
        name="test_dataset",
        splits=[],
    )

    with pytest.raises(
        ValueError,
        match="image_width must be greater than 0.",
    ):
        VisionDetectionDataset_to_VisionYoloDataset(
            dataset=dataset,
            image_width=0,
            image_height=500,
        )


def test_raise_when_image_height_is_zero():

    dataset = VisionDetectionDataset(
        name="test_dataset",
        splits=[],
    )

    with pytest.raises(
        ValueError,
        match="image_height must be greater than 0.",
    ):
        VisionDetectionDataset_to_VisionYoloDataset(
            dataset=dataset,
            image_width=1000,
            image_height=0,
        )


def test_raise_when_sample_target_is_none():

    dataset = VisionDetectionDataset(
        name="test_dataset",
        splits=[
            DatasetSplit(
                name="train",
                samples=[
                    Sample(
                        input="train_001.jpg",
                        target=None,
                    ),
                ],
            ),
        ],
    )

    with pytest.raises(
        ValueError,
        match="sample.target must not be None.",
    ):
        VisionDetectionDataset_to_VisionYoloDataset(
            dataset=dataset,
            image_width=1000,
            image_height=500,
        )