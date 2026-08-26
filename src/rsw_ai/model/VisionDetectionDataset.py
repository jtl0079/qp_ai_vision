from rsw_ai.model.VisionDataset import VisionDataset
from rsw_ai.model.DetectionObject import DetectionObject

"""
VisionDetectionDataset
│
├── name : str
│
├── class_map : ClassMap
│   │
│   ├── 0 -> "car"
│   ├── 1 -> "person"
│   ├── 2 -> "bicycle"
│   └── ...
│
└── splits : list<DatasetSplit>
    │
    ├── DatasetSplit ("train")
    │   │
    │   ├── name : str
    │   │
    │   └── samples : list<Sample>
    │       │
    │       ├── Sample
    │       │   │
    │       │   ├── input : str
    │       │   │      "000001.jpg"
    │       │   │
    │       │   └── target : list<DetectionObject>
    │       │       │
    │       │       ├── DetectionObject
    │       │       │   │
    │       │       │   ├── class_id : int
    │       │       │   ├── x : float
    │       │       │   ├── y : float
    │       │       │   ├── width : float
    │       │       │   └── height : float
    │       │       │
    │       │       ├── DetectionObject
    │       │       │   │
    │       │       │   ├── class_id : int
    │       │       │   ├── x : float
    │       │       │   ├── y : float
    │       │       │   ├── width : float
    │       │       │   └── height : float
    │       │       │
    │       │       └── ...
    │       │
    │       ├── Sample
    │       │   ├── input : str
    │       │   └── target : list<DetectionObject>
    │       │
    │       └── ...
    │
    ├── DatasetSplit ("val")
    │   └── samples : list<Sample>
    │
    └── DatasetSplit ("test")
        └── samples : list<Sample>

"""


class VisionDetectionDataset(VisionDataset[list[DetectionObject]]):
    pass
