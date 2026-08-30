def yolo_prediction_function(
    model,
    image,
    device
):

    # YOLO inference
    results = model(
        image,
        verbose=False
    )[0]

    # YOLO output
    boxes = results.boxes.xyxy.cpu()

    labels = (
        results.boxes.cls
        .cpu()
        .long()
        + 1
    )

    scores = results.boxes.conf.cpu()

    # Return the same format required
    # by ObjectDetectionEvaluator
    return {
        "boxes": boxes,
        "labels": labels,
        "scores": scores
    }