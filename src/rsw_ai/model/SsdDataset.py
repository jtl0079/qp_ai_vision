from dataclasses import dataclass, field
from .SsdAnnotation import SsdAnnotation

@dataclass
class SsdDataset:
    classes: list[str] = field(default_factory=list)
    annotations: dict[str, SsdAnnotation] = field(default_factory=dict)
    """
      classes structure

        classes
        └── list of class (list)
            [0] Car
            [1] Threewheel
            [2] Bus
            ...


      annotation structure

        annotation
        └──filename (str)
           └── each annotation 
               └── bbox location
                   class (list int)
                   width (int)
                   height (int)


        Example
        -------
        {
            "image1.jpg": annotation for image1
        }
    """

