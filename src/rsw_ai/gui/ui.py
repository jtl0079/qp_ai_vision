import cv2
import torch
import torchvision
import numpy as np
import gradio as gr

import torchvision.transforms.functional as TF


# ============================================================
# 1. DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("Using device:", device)


# ============================================================
# 2. CLASS NAMES
# ============================================================

class_names = [
    "CAR",
    "THREEWHEEL",
    "BUS",
    "TRUCK",
    "MOTORBIKE",
    "VAN"
]


# ============================================================
# 3. LOAD SSD
# ============================================================

ssd_model = torchvision.models.detection.ssd300_vgg16(
    num_classes=7
)

ssd_model.load_state_dict(
    torch.load(
        "/content/drive/MyDrive/RSW_Y2S1_AI/dataset/ssd_model.pth",
        map_location=device,
        weights_only=True
    )
)

ssd_model.to(device)
ssd_model.eval()

print("SSD model loaded.")


# ============================================================
# 4. YOLO
# ============================================================

yolo_model = None

# Example:
#
# from ultralytics import YOLO
#
# yolo_model = YOLO(
#     "/content/drive/MyDrive/RSW_Y2S1_AI/best.pt"
# )


# ============================================================
# 5. MASK R-CNN
# ============================================================

mask_rcnn_model = None

# Example:
#
# mask_rcnn_model = ...
#
# mask_rcnn_model.to(device)
# mask_rcnn_model.eval()


# ============================================================
# 6. SSD PREDICTION
# ============================================================

def predict_ssd(image, confidence_threshold):

    image_tensor = TF.to_tensor(image)

    image_tensor = image_tensor.to(device)

    with torch.no_grad():

        prediction = ssd_model(
            [image_tensor]
        )[0]

    boxes = prediction["boxes"].cpu()
    labels = prediction["labels"].cpu()
    scores = prediction["scores"].cpu()

    result_image = image.copy()

    detected_objects = []

    for box, label, score in zip(
        boxes,
        labels,
        scores
    ):

        score = score.item()

        if score < confidence_threshold:
            continue

        class_id = label.item() - 1

        if class_id < 0:
            continue

        if class_id >= len(class_names):
            continue

        class_name = class_names[class_id]

        x_min, y_min, x_max, y_max = map(
            int,
            box.tolist()
        )

        cv2.rectangle(
            result_image,
            (x_min, y_min),
            (x_max, y_max),
            (0, 255, 0),
            2
        )

        cv2.putText(
            result_image,
            f"{class_name}: {score:.2f}",
            (
                x_min,
                max(y_min - 10, 20)
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

        detected_objects.append(
            f"{class_name}: {score:.2f}"
        )

    return (
        result_image,
        detected_objects
    )


# ============================================================
# 7. YOLO PREDICTION
# ============================================================

def predict_yolo(
    image,
    confidence_threshold
):

    if yolo_model is None:

        return (
            image,
            ["YOLO model has not been loaded."]
        )

    # --------------------------------------------------------
    # YOLO prediction
    # --------------------------------------------------------

    # Example:
    #
    # results = yolo_model(
    #     image,
    #     conf=confidence_threshold
    # )
    #
    # result_image = results[0].plot()
    #
    # detected_objects = []
    #
    # for box in results[0].boxes:
    #
    #     class_id = int(box.cls[0])
    #     confidence = float(box.conf[0])
    #
    #     class_name = class_names[class_id]
    #
    #     detected_objects.append(
    #         f"{class_name}: {confidence:.2f}"
    #     )

    return (
        image,
        ["YOLO prediction"]
    )


# ============================================================
# 8. MASK R-CNN PREDICTION
# ============================================================

def predict_mask_rcnn(
    image,
    confidence_threshold
):

    if mask_rcnn_model is None:

        return (
            image,
            ["Mask R-CNN model has not been loaded."]
        )

    # --------------------------------------------------------
    # Mask R-CNN prediction
    # --------------------------------------------------------

    # Future implementation

    return (
        image,
        ["Mask R-CNN prediction"]
    )


# ============================================================
# 9. SINGLE MODEL
# ============================================================

def predict_single(
    image,
    model_name,
    confidence_threshold
):

    if image is None:

        return (
            None,
            "Please upload an image."
        )

    # ========================================================
    # SSD
    # ========================================================

    if model_name == "SSD":

        result_image, objects = predict_ssd(
            image,
            confidence_threshold
        )

    # ========================================================
    # YOLO
    # ========================================================

    elif model_name == "YOLO":

        result_image, objects = predict_yolo(
            image,
            confidence_threshold
        )

    # ========================================================
    # MASK R-CNN
    # ========================================================

    elif model_name == "Mask R-CNN":

        result_image, objects = predict_mask_rcnn(
            image,
            confidence_threshold
        )

    else:

        return (
            image,
            "Unknown model."
        )

    # ========================================================
    # RESULT TEXT
    # ========================================================

    if len(objects) == 0:

        result_text = "No objects detected."

    else:

        result_text = "\n".join(objects)

    return (
        result_image,
        result_text
    )


# ============================================================
# 10. EVALUATION FUNCTIONS
# ============================================================

def calculate_precision(
    true_positive,
    false_positive
):

    denominator = (
        true_positive +
        false_positive
    )

    if denominator == 0:
        return 0.0

    return true_positive / denominator


def calculate_recall(
    true_positive,
    false_negative
):

    denominator = (
        true_positive +
        false_negative
    )

    if denominator == 0:
        return 0.0

    return true_positive / denominator


def calculate_f1(
    precision,
    recall
):

    denominator = precision + recall

    if denominator == 0:
        return 0.0

    return (
        2 *
        precision *
        recall /
        denominator
    )


# ============================================================
# 11. IOU
# ============================================================

def calculate_iou(
    box1,
    box2
):

    x1 = max(
        box1[0],
        box2[0]
    )

    y1 = max(
        box1[1],
        box2[1]
    )

    x2 = min(
        box1[2],
        box2[2]
    )

    y2 = min(
        box1[3],
        box2[3]
    )

    intersection_width = max(
        0,
        x2 - x1
    )

    intersection_height = max(
        0,
        y2 - y1
    )

    intersection = (
        intersection_width *
        intersection_height
    )

    area1 = (
        (box1[2] - box1[0]) *
        (box1[3] - box1[1])
    )

    area2 = (
        (box2[2] - box2[0]) *
        (box2[3] - box2[1])
    )

    union = (
        area1 +
        area2 -
        intersection
    )

    if union == 0:
        return 0.0

    return intersection / union


# ============================================================
# 12. MODEL EVALUATION PLACEHOLDER
# ============================================================

def evaluate_model(
    model_name
):

    # --------------------------------------------------------
    # IMPORTANT
    # --------------------------------------------------------
    #
    # Real Precision / Recall / F1 / mAP require:
    #
    # 1. Validation images
    # 2. Ground-truth bounding boxes
    # 3. Ground-truth class labels
    #
    # They cannot be calculated correctly from only
    # one uploaded image.
    #
    # --------------------------------------------------------

    if model_name == "SSD":

        # ----------------------------------------------------
        # Placeholder
        # ----------------------------------------------------
        #
        # Replace these values with results calculated
        # from your validation dataset.
        #
        precision = 0.0
        recall = 0.0
        f1 = 0.0
        map50 = 0.0
        map5095 = 0.0

    elif model_name == "YOLO":

        precision = 0.0
        recall = 0.0
        f1 = 0.0
        map50 = 0.0
        map5095 = 0.0

    elif model_name == "Mask R-CNN":

        precision = 0.0
        recall = 0.0
        f1 = 0.0
        map50 = 0.0
        map5095 = 0.0

    else:

        precision = 0.0
        recall = 0.0
        f1 = 0.0
        map50 = 0.0
        map5095 = 0.0

    return {
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
        "mAP@0.5": map50,
        "mAP@0.5:0.95": map5095
    }


# ============================================================
# 13. COMPARE MODELS
# ============================================================

def compare_models():

    # ========================================================
    # Get evaluation results
    # ========================================================

    ssd = evaluate_model(
        "SSD"
    )

    yolo = evaluate_model(
        "YOLO"
    )

    mask_rcnn = evaluate_model(
        "Mask R-CNN"
    )

    # ========================================================
    # Comparison table
    # ========================================================

    comparison = f"""
MODEL PERFORMANCE COMPARISON
=================================================================================================================================================================

Metric                  SSD          YOLO        Mask R-CNN
-----------------------------------------------------------------------------------------------------------------------------------------------------------------

Precision               {ssd["Precision"]:.4f}       {yolo["Precision"]:.4f}       {mask_rcnn["Precision"]:.4f}

Recall                  {ssd["Recall"]:.4f}       {yolo["Recall"]:.4f}       {mask_rcnn["Recall"]:.4f}

F1-score                {ssd["F1-score"]:.4f}       {yolo["F1-score"]:.4f}       {mask_rcnn["F1-score"]:.4f}

mAP@0.5                 {ssd["mAP@0.5"]:.4f}       {yolo["mAP@0.5"]:.4f}       {mask_rcnn["mAP@0.5"]:.4f}

mAP@0.5:0.95            {ssd["mAP@0.5:0.95"]:.4f}       {yolo["mAP@0.5:0.95"]:.4f}       {mask_rcnn["mAP@0.5:0.95"]:.4f}


================================================================================================================================================================
INTERPRETATION

Precision:
Measures how many detected objects are actually correct.

Recall:
Measures how many actual objects were successfully detected.

F1-score:
Balances Precision and Recall.

mAP@0.5:
Measures object detection performance using IoU = 0.50.

mAP@0.5:0.95:
Measures detection performance across multiple IoU thresholds.

Higher values indicate better detection performance.
"""

    return comparison


# ============================================================
# 14. GRADIO UI
# ============================================================

with gr.Blocks() as demo:

    # ========================================================
    # TITLE
    # ========================================================

    gr.Markdown(
        """
        # AI Vehicle Detection & Model Comparison

        Compare SSD, YOLO and Mask R-CNN.
        """
    )

    # ========================================================
    # INPUT IMAGE
    # ========================================================

    input_image = gr.Image(
        type="numpy",
        label="Upload Image"
    )

    # ========================================================
    # CONFIDENCE
    # ========================================================

    confidence_slider = gr.Slider(
        minimum=0.1,
        maximum=1.0,
        value=0.65,
        step=0.05,
        label="Confidence Threshold"
    )

    # ========================================================
    # SINGLE MODEL
    # ========================================================

    gr.Markdown(
        "## Single Model Detection"
    )

    model_dropdown = gr.Dropdown(
        choices=[
            "SSD",
            "YOLO",
            "Mask R-CNN"
        ],
        value="SSD",
        label="Select Model"
    )

    detect_button = gr.Button(
        "Detect",
        variant="primary"
    )

    single_output_image = gr.Image(
        label="Prediction Result"
    )

    single_output_text = gr.Textbox(
        label="Detection Result",
        lines=10
    )

    detect_button.click(
        fn=predict_single,

        inputs=[
            input_image,
            model_dropdown,
            confidence_slider
        ],

        outputs=[
            single_output_image,
            single_output_text
        ]
    )

    # ========================================================
    # MODEL COMPARISON
    # ========================================================

    gr.Markdown(
        "## Model Performance Comparison"
    )

    gr.Markdown(
        """
        The comparison uses Precision, Recall, F1-score,
        mAP@0.5 and mAP@0.5:0.95.
        """
    )

    compare_button = gr.Button(
        "Compare Model Performance",
        variant="primary"
    )

    comparison_output = gr.Textbox(
        label="Performance Comparison",
        lines=25
    )

    compare_button.click(
        fn=compare_models,

        inputs=[],

        outputs=[
            comparison_output
        ]
    )


# ============================================================
# 15. LAUNCH
# ============================================================

demo.launch(
    share=True
)