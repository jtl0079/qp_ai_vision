from dataclasses import dataclass, field

@dataclass
class SsdAnnotation:
    boxes: list[list[float]] = field(default_factory=list)
    labels: list[int] = field(default_factory=list)
    width: int = 0
    height: int = 0

    """
      boxes structure
        boxes
        └── list of box(list)
            └── each box (list)
                └── [0] x_min
                    [1] y_min
                    [2] x_max
                    [3] y_max
        Example
        -------
        {
            [
              box 1 = [10.1,20.2,30.3,40.4],
              box 2 = [12.1，15.1，30.11，90.6],
              ....
            ]
        }


      labels structure
        labels
        └── list of label(list)
            └── [0] object 1 label(int)
                [1] object 2 label(int)
                [2] object 3 label(int)
                [3] object 4 label(int)
        Example
        -------
        {
            [
              1,2,.....
            ]
        }
    """
    def set_width(self,width):
      self.width = width

    def set_height(self,height):
      self.height = height