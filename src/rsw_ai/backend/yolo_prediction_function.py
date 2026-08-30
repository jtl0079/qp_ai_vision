import torch
import torch.nn.functional as F


def yolo_prediction_function(
    model,
    image,
    device
):

    # ========================================================
    # Prepare image
    # ========================================================

    if not isinstance(image, torch.Tensor):
        image = torch.from_numpy(image)

    # --------------------------------------------------------
    # C,H,W -> 1,C,H,W
    # --------------------------------------------------------

    if image.ndim == 3:
        image = image.unsqueeze(0)

    # --------------------------------------------------------
    # Make sure float
    # --------------------------------------------------------

    image = image.float()

    # --------------------------------------------------------
    # If image is 0-255, convert to 0-1
    # --------------------------------------------------------

    if image.max() > 1.0:
        image = image / 255.0

    # --------------------------------------------------------
    # BGR -> RGB
    # --------------------------------------------------------

    image = image[:, [2, 1, 0], :, :]

    # --------------------------------------------------------
    # YOLO input size
    #
    # 320 is divisible by 32
    # --------------------------------------------------------

    original_height = image.shape[2]
    original_width = image.shape[3]

    image = F.interpolate(
        image,
        size=(320, 320),
        mode="bilinear",
        align_corners=False
    )

    # --------------------------------------------------------
    # Move to device
    # --------------------------------------------------------

    image = image.to(device)

    # ========================================================
    # YOLO inference
    # ========================================================

    with torch.no_grad():

        results = model.predict(
            source=image,
            verbose=False,
            conf=0.0
        )[0]

    # ========================================================
    # No detection
    # ========================================================

    if (
        results.boxes is None
        or
        len(results.boxes) == 0
    ):

        return {
            "boxes": torch.empty(
                (0, 4),
                dtype=torch.float32
            ),

            "labels": torch.empty(
                (0,),
                dtype=torch.long
            ),

            "scores": torch.empty(
                (0,),
                dtype=torch.float32
            )
        }

    # ========================================================
    # Get predictions
    # ========================================================

    boxes = results.boxes.xyxy.cpu()

    labels = (
        results.boxes.cls
        .cpu()
        .long()
        + 1
    )

    scores = (
        results.boxes.conf
        .cpu()
    )

    # ========================================================
    # Scale boxes
    #
    # YOLO boxes are currently based on 320x320.
    # Evaluator ground truth is based on 300x300.
    # ========================================================

    scale_x = (
        original_width / 320.0
    )

    scale_y = (
        original_height / 320.0
    )

    boxes[:, 0] *= scale_x
    boxes[:, 2] *= scale_x

    boxes[:, 1] *= scale_y
    boxes[:, 3] *= scale_y

    # ========================================================
    # Return evaluator format
    # ========================================================

    return {
        "boxes": boxes,
        "labels": labels,
        "scores": scores
    }