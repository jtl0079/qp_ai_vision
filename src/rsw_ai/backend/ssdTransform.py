import cv2
import torch
import numpy as np
import torchvision.transforms.functional as TF


class SsdTransform:

    def __init__(
        self,
        resize=None,

        horizontal_flip=False,
        vertical_flip=False,

        grayscale=False,
        binary=False,
        morphology=None,
        denoise=None,

        brightness=None,
        contrast=None,

        normalize=False,
    ):
        self.resize = resize
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.grayscale = grayscale
        self.binary = binary
        self.morphology = morphology
        self.denoise = denoise
        self.brightness = brightness
        self.contrast = contrast
        self.normalize = normalize

    def __call__(self, image, target):

        # =====================================
        # Resize
        # =====================================

        if self.resize is not None:

            new_width, new_height = self.resize

            old_height, old_width = image.shape[:2]

            image = cv2.resize(
                image,
                (new_width, new_height)
            )

            boxes = target["boxes"].clone()

            boxes[:, [0, 2]] *= new_width / old_width
            boxes[:, [1, 3]] *= new_height / old_height

            target["boxes"] = boxes

        # =====================================
        # Horizontal Flip
        # =====================================

        if self.horizontal_flip:

            image = cv2.flip(image, 1)

            width = image.shape[1]

            boxes = target["boxes"].clone()

            old_x_min = boxes[:, 0].clone()
            old_x_max = boxes[:, 2].clone()

            boxes[:, 0] = width - old_x_max
            boxes[:, 2] = width - old_x_min

            target["boxes"] = boxes

        # =====================================
        # Vertical Flip
        # =====================================

        if self.vertical_flip:

            image = cv2.flip(image, 0)

            height = image.shape[0]

            boxes = target["boxes"].clone()

            old_y_min = boxes[:, 1].clone()
            old_y_max = boxes[:, 3].clone()

            boxes[:, 1] = height - old_y_max
            boxes[:, 3] = height - old_y_min

            target["boxes"] = boxes

        # =====================================
        # Brightness
        # =====================================

        if self.brightness is not None:

            image = np.clip(
                image.astype(np.float32)
                * self.brightness,
                0,
                255
            ).astype(np.uint8)

        # =====================================
        # Contrast
        # =====================================

        if self.contrast is not None:

            mean = np.mean(
                image,
                axis=(0, 1),
                keepdims=True
            )

            image = np.clip(
                (
                    image.astype(np.float32)
                    - mean
                )
                * self.contrast
                + mean,
                0,
                255
            ).astype(np.uint8)

        # =====================================
        # Grayscale
        # =====================================

        if self.grayscale:

          image = cv2.cvtColor(
              image,
              cv2.COLOR_RGB2GRAY
          )

          image = cv2.cvtColor(
              image,
              cv2.COLOR_GRAY2RGB
          )

        # =====================================
        # Binary
        # =====================================

        if self.binary:

          gray = cv2.cvtColor(
              image,
              cv2.COLOR_RGB2GRAY
          )

          _, binary_image = cv2.threshold(
              gray,
              127,
              255,
              cv2.THRESH_BINARY
          )

          image = cv2.cvtColor(
              binary_image,
              cv2.COLOR_GRAY2RGB
          )

        # =====================================
        # Morphological Operation
        # =====================================

        if self.morphology == "erosion":

          kernel = np.ones(
          (3, 3),
          np.uint8
          )

          image = cv2.erode(
              image,
              kernel,
              iterations=1
          )


        elif self.morphology == "dilation":

          kernel = np.ones(
              (3, 3),
              np.uint8
          )

          image = cv2.dilate(
              image,
              kernel,
              iterations=1
          )


        elif self.morphology == "opening":

          kernel = np.ones(
              (3, 3),
              np.uint8
          )

          image = cv2.morphologyEx(
              image,
              cv2.MORPH_OPEN,
              kernel
          )


        elif self.morphology == "closing":

          kernel = np.ones(
              (3, 3),
              np.uint8
          )

          image = cv2.morphologyEx(
              image,
              cv2.MORPH_CLOSE,
              kernel
          )


        # =====================================
        # Denoise
        # =====================================

        if self.denoise == "average":
          image = cv2.blur(
              image,
              (7, 7)
        )


        elif self.denoise == "gaussian":
          image = cv2.GaussianBlur(
              image,
              (5, 5),
              0
          )


        elif self.denoise == "median":
          image = cv2.medianBlur(
              image,
              3
          )

        # =====================================
        # Image → Tensor
        # =====================================

        image = TF.to_tensor(image)

        # =====================================
        # Normalize
        # =====================================

        if self.normalize:

            image = TF.normalize(
                image,
                mean=[
                    0.485,
                    0.456,
                    0.406
                ],
                std=[
                    0.229,
                    0.224,
                    0.225
                ]
            )

        return image, target