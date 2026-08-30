import cv2
import torch
import torchvision
import numpy as np
import gradio as gr
import pickle
import sys

from ultralytics import YOLO

sys.path.insert(0, "/content/qp_ai_vision/src")
from rsw_ai.backend.ssdTransform import SsdTransform
from rsw_ai.model.SsdTorchDataset import SsdTorchDataset

from rsw_ai.model.ObjectDetectionEvaluator import (
    ObjectDetectionEvaluator
)
from rsw_ai.backend.yolo_prediction_function import yolo_prediction_function


# ============================================================
# 1. DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
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
# 3. EVALUATOR
# ============================================================

evaluator = ObjectDetectionEvaluator(
    iou_threshold=0.5,
    confidence_threshold=0.65
)


# ============================================================
# 4. LOAD VALIDATION DATASET
# ============================================================

with open(
    "/content/drive/MyDrive/"
    "RSW_Y2S1_AI/dataset/ssd_data.pkl",
    "rb"
) as f:

    ssd_dataset = pickle.load(f)


valid_transform = SsdTransform(
    resize=(300, 300),
    horizontal_flip=False,
    vertical_flip=False,
    denoise="gaussian",
    brightness=1.0,
    contrast=1.5,
    normalize=False
)


valid_dataset = SsdTorchDataset(
    ssd_dataset=ssd_dataset,
    transform=valid_transform,
    split="valid"
)


print(
    "Validation images:",
    len(valid_dataset)
)


# ============================================================
# 5. LOAD SSD MODEL
# ============================================================

ssd_model = torchvision.models.detection.ssd300_vgg16(
    num_classes=7,
    score_thresh=0.0,
    nms_thresh=0.3,
    detections_per_img=100
)


"\outputs\models\ssd\ssd_model.pth"
ssd_model.load_state_dict(
    torch.load(
        "/content/drive/MyDrive/"
        "RSW_Y2S1_AI/dataset/ssd_model/"
        "ssd_best_model.pth",
        map_location=device,
        weights_only=True
    )
)


ssd_model.to(device)
ssd_model.eval()


print("SSD model loaded.")


# ============================================================
# 6. YOLO MODEL
# ============================================================

yolo_model = None

yolo_model = YOLO(
    "../outputs/experiments/yolo/nadinpethiyagoda_vehicle_dataset_for_yolo/weights/best.pt"
)

# Example:
#
# from ultralytics import YOLO
#
# yolo_model = YOLO(
#     "/content/drive/MyDrive/RSW_Y2S1_AI/best.pt"
# )


# ============================================================
# 7. MASK R-CNN MODEL
# ============================================================

mask_rcnn_model = None


# Example:
#
# mask_rcnn_model = ...
#
# mask_rcnn_model.to(device)
# mask_rcnn_model.eval()


# ============================================================
# 8. SSD PREDICTION FUNCTION
# ============================================================

def ssd_prediction_function(
    model,
    image,
    device
):

    image = image.to(device)


    with torch.no_grad():

        prediction = model(
            [image]
        )[0]


    return {

        "boxes":
            prediction["boxes"].cpu(),

        "labels":
            prediction["labels"].cpu(),

        "scores":
            prediction["scores"].cpu()

    }


# ============================================================
# 9. SSD GRADIO DETECTION
# ============================================================

def predict_ssd(
    image,
    confidence_threshold
):

    if image is None:

        return (
            None,
            []
        )

    # --------------------------------------------------------
    # Keep ORIGINAL user image
    # --------------------------------------------------------

    original_image = image.copy()

    original_height, original_width = (
        original_image.shape[:2]
    )

    # --------------------------------------------------------
    # Resize image for SSD
    # SSD300 requires 300 x 300 input
    # --------------------------------------------------------

    transformed = cv2.resize(
        original_image,
        (300, 300)
    )

    # --------------------------------------------------------
    # Gaussian denoise
    # Same preprocessing as validation
    # --------------------------------------------------------

    transformed = cv2.GaussianBlur(
        transformed,
        (5, 5),
        0
    )

    # --------------------------------------------------------
    # Contrast = 1.5
    # --------------------------------------------------------

    mean = np.mean(
        transformed,
        axis=(0, 1),
        keepdims=True
    )

    transformed = np.clip(
        (
            transformed.astype(
                np.float32
            )
            - mean
        )
        * 1.5
        + mean,
        0,
        255
    ).astype(
        np.uint8
    )

    # --------------------------------------------------------
    # Convert to Tensor
    # 0-255 -> 0-1
    # --------------------------------------------------------

    tensor_image = (
        torch.from_numpy(
            transformed
        )
        .permute(2, 0, 1)
        .float()
        / 255.0
    )

    tensor_image = tensor_image.to(
        device
    )

    # --------------------------------------------------------
    # SSD Prediction
    # --------------------------------------------------------

    prediction = ssd_prediction_function(
        ssd_model,
        tensor_image,
        device
    )

    boxes = prediction["boxes"]
    labels = prediction["labels"]
    scores = prediction["scores"]

    # --------------------------------------------------------
    # Confidence filtering
    # --------------------------------------------------------

    keep = (
        scores >= confidence_threshold
    )

    boxes = boxes[keep]
    labels = labels[keep]
    scores = scores[keep]

    # --------------------------------------------------------
    # Draw on ORIGINAL user image
    # --------------------------------------------------------

    result_image = original_image.copy()

    detected_objects = []

    # --------------------------------------------------------
    # Scale bounding boxes
    # from 300 x 300
    # back to ORIGINAL image size
    # --------------------------------------------------------

    scale_x = (
        original_width / 300.0
    )

    scale_y = (
        original_height / 300.0
    )

    # --------------------------------------------------------
    # Draw predictions
    # --------------------------------------------------------

    for box, label, score in zip(
        boxes,
        labels,
        scores
    ):

        x1, y1, x2, y2 = (
            box.tolist()
        )

        # Scale back to original image
        x1 = int(x1 * scale_x)
        y1 = int(y1 * scale_y)
        x2 = int(x2 * scale_x)
        y2 = int(y2 * scale_y)

        # ----------------------------------------------------
        # SSD label:
        # 0 = background
        # 1 = CAR
        # 2 = THREEWHEEL
        # ...
        # 6 = VAN
        #
        # Convert to class_names index
        # ----------------------------------------------------

        class_id = (
            int(label.item()) - 1
        )

        if (
            class_id < 0
            or class_id >= len(class_names)
        ):

            continue

        class_name = class_names[
            class_id
        ]

        confidence = float(
            score.item()
        )

        # ----------------------------------------------------
        # Draw bounding box
        # ----------------------------------------------------

        cv2.rectangle(
            result_image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        # ----------------------------------------------------
        # Draw label
        # ----------------------------------------------------

        text = (
            f"{class_name}: "
            f"{confidence:.2f}"
        )

        cv2.putText(
            result_image,
            text,
            (
                x1,
                max(y1 - 10, 20)
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )

        detected_objects.append(
            text
        )

    # --------------------------------------------------------
    # Return ORIGINAL SIZE image
    # with bounding boxes
    # --------------------------------------------------------

    return (
        result_image,
        detected_objects
    )

# ============================================================
# 10. YOLO PREDICTION
# ============================================================

def predict_yolo(
    image,
    confidence_threshold
):

    if yolo_model is None:

        return (
            image,
            [
                "YOLO model has not been loaded."
            ]
        )


    # Future YOLO implementation


    # --------------------------------------------------------
    # YOLO inference
    # --------------------------------------------------------

    results = yolo_model.predict(
        source=image,
        conf=confidence_threshold,
        verbose=False
    )[0]

    # --------------------------------------------------------
    # Keep original image
    # --------------------------------------------------------

    result_image = image.copy()

    detected_objects = []

    # --------------------------------------------------------
    # Get predictions
    # --------------------------------------------------------

    boxes = results.boxes.xyxy.cpu().numpy()
    labels = results.boxes.cls.cpu().numpy().astype(int)
    scores = results.boxes.conf.cpu().numpy()

    # --------------------------------------------------------
    # Draw predictions
    # --------------------------------------------------------

    for box, label, score in zip(
        boxes,
        labels,
        scores
    ):

        x1, y1, x2, y2 = box.astype(int)

        # YOLO class ID:
        #
        # 0 = CAR
        # 1 = THREEWHEEL
        # 2 = BUS
        # 3 = TRUCK
        # 4 = MOTORBIKE
        # 5 = VAN

        if (
            label < 0
            or label >= len(class_names)
        ):
            continue

        class_name = class_names[label]

        confidence = float(score)

        # ----------------------------------------------------
        # Bounding box
        # ----------------------------------------------------

        cv2.rectangle(
            result_image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        # ----------------------------------------------------
        # Label
        # ----------------------------------------------------

        text = (
            f"{class_name}: "
            f"{confidence:.2f}"
        )

        cv2.putText(
            result_image,
            text,
            (
                x1,
                max(y1 - 10, 20)
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )

        detected_objects.append(text)

    return (
        result_image,
        detected_objects
    )


# ============================================================
# 11. MASK R-CNN PREDICTION
# ============================================================

def predict_mask_rcnn(
    image,
    confidence_threshold
):

    if mask_rcnn_model is None:

        return (
            image,
            [
                "Mask R-CNN model has not been loaded."
            ]
        )


    # Future Mask R-CNN implementation


    return (
        image,
        [
            "Mask R-CNN prediction"
        ]
    )


# ============================================================
# 12. SINGLE MODEL DETECTION
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

        result_image, objects = (
            predict_ssd(
                image,
                confidence_threshold
            )
        )


    # ========================================================
    # YOLO
    # ========================================================

    elif model_name == "YOLO":

        result_image, objects = (
            predict_yolo(
                image,
                confidence_threshold
            )
        )


    # ========================================================
    # MASK R-CNN
    # ========================================================

    elif model_name == "Mask R-CNN":

        result_image, objects = (
            predict_mask_rcnn(
                image,
                confidence_threshold
            )
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

        result_text = (
            "No objects detected."
        )

    else:

        result_text = "\n".join(
            objects
        )


    return (
        result_image,
        result_text
    )


# ============================================================
# 13. EVALUATE MODEL
# ============================================================

def evaluate_model(
    model_name
):

    # ========================================================
    # SSD
    # ========================================================

    if model_name == "SSD":

        results = evaluator.evaluate(

            model=ssd_model,

            dataset=valid_dataset,

            device=device,

            prediction_function=(
                ssd_prediction_function
            )

        )

        return results


    # ========================================================
    # YOLO
    # ========================================================

    elif model_name == "YOLO":

        results = evaluator.evaluate(
            model=yolo_model,
            dataset=valid_dataset,
            device=device,
            prediction_function=(
                yolo_prediction_function
            )
        )

        return results



    # ========================================================
    # MASK R-CNN
    # ========================================================

    elif model_name == "Mask R-CNN":

        return {

            "Precision": 0.0,

            "Recall": 0.0,

            "F1-score": 0.0,

            "IoU": 0.0,

            "mAP@0.5": 0.0,

            "mAP@0.5:0.95": 0.0

        }


    return {}


# ============================================================
# 14. COMPARE MODELS
# ============================================================

def compare_models():

    ssd = evaluate_model(
        "SSD"
    )


    yolo = evaluate_model(
        "YOLO"
    )


    mask_rcnn = evaluate_model(
        "Mask R-CNN"
    )


    comparison = f"""

MODEL PERFORMANCE COMPARISON
==============================================================

Metric                  SSD          YOLO        Mask R-CNN
--------------------------------------------------------------

Precision               {ssd["Precision"]:.4f}       {yolo["Precision"]:.4f}       {mask_rcnn["Precision"]:.4f}

Recall                  {ssd["Recall"]:.4f}       {yolo["Recall"]:.4f}       {mask_rcnn["Recall"]:.4f}

F1-score                {ssd["F1-score"]:.4f}       {yolo["F1-score"]:.4f}       {mask_rcnn["F1-score"]:.4f}

IoU                     {ssd["IoU"]:.4f}       {yolo["IoU"]:.4f}       {mask_rcnn["IoU"]:.4f}

mAP@0.5                 {ssd["mAP@0.5"]:.4f}       {yolo["mAP@0.5"]:.4f}       {mask_rcnn["mAP@0.5"]:.4f}

mAP@0.5:0.95            {ssd["mAP@0.5:0.95"]:.4f}       {yolo["mAP@0.5:0.95"]:.4f}       {mask_rcnn["mAP@0.5:0.95"]:.4f}


==============================================================

INTERPRETATION

Precision:
Of all predicted objects, how many are correct.

Recall:
Of all actual objects, how many are detected.

F1-score:
Harmonic mean of Precision and Recall.

IoU:
Measures overlap between predicted bounding boxes
and ground-truth bounding boxes.

mAP@0.5:
Mean Average Precision at IoU = 0.50.

mAP@0.5:0.95:
Mean Average Precision from IoU 0.50 to 0.95.

Higher values indicate better detection performance.

"""


    return comparison


# ============================================================
# 15. GRADIO UI
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
        The comparison uses Precision, Recall,
        F1-score, IoU, mAP@0.5 and mAP@0.5:0.95.
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
# 16. LAUNCH
# ============================================================

demo.launch(
    share=True
)