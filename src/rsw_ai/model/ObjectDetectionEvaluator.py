import torch
import numpy as np


class ObjectDetectionEvaluator:

    def __init__(
        self,
        iou_threshold=0.5,
        confidence_threshold=0.65,
        map_iou_thresholds=None
    ):

        self.iou_threshold = iou_threshold
        self.confidence_threshold = confidence_threshold

        if map_iou_thresholds is None:

            self.map_iou_thresholds = np.arange(0.50,0.96,0.05)

        else:

            self.map_iou_thresholds = np.array(
                map_iou_thresholds
            )

    # =========================================================
    # IoU
    # =========================================================

    def calculate_iou(
        self,
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
            max(
                0,
                box1[2] - box1[0]
            )
            *
            max(
                0,
                box1[3] - box1[1]
            )
        )

        area2 = (
            max(
                0,
                box2[2] - box2[0]
            )
            *
            max(
                0,
                box2[3] - box2[1]
            )
        )

        union = (
            area1 +
            area2 -
            intersection
        )

        if union <= 0:

            return 0.0

        return float(
            intersection / union
        )

    # =========================================================
    # Match predictions
    #
    # Used for:
    # Precision
    # Recall
    # F1
    # =========================================================

    def match_predictions(
        self,
        prediction_boxes,
        prediction_labels,
        prediction_scores,
        ground_truth_boxes,
        ground_truth_labels,
        iou_threshold
    ):

        if len(prediction_boxes) == 0:

            return (
                0,
                0,
                len(ground_truth_boxes)
            )

        if len(ground_truth_boxes) == 0:

            return (
                0,
                len(prediction_boxes),
                0
            )

        prediction_boxes = (
            prediction_boxes.cpu().numpy()
        )

        prediction_labels = (
            prediction_labels.cpu().numpy()
        )

        prediction_scores = (
            prediction_scores.cpu().numpy()
        )

        ground_truth_boxes = (
            ground_truth_boxes.cpu().numpy()
        )

        ground_truth_labels = (
            ground_truth_labels.cpu().numpy()
        )

        # -----------------------------------------------------
        # Sort predictions by confidence
        # -----------------------------------------------------

        order = np.argsort(
            -prediction_scores
        )

        prediction_boxes = (
            prediction_boxes[order]
        )

        prediction_labels = (
            prediction_labels[order]
        )

        matched_ground_truth = set()

        true_positive = 0
        false_positive = 0

        # -----------------------------------------------------
        # Match
        # -----------------------------------------------------

        for pred_box, pred_label in zip(
            prediction_boxes,
            prediction_labels
        ):

            best_iou = 0.0
            best_gt_index = -1

            for gt_index, (
                gt_box,
                gt_label
            ) in enumerate(
                zip(
                    ground_truth_boxes,
                    ground_truth_labels
                )
            ):

                if gt_index in matched_ground_truth:
                    continue

                if pred_label != gt_label:
                    continue

                iou = self.calculate_iou(
                    pred_box,
                    gt_box
                )

                if iou > best_iou:

                    best_iou = iou
                    best_gt_index = gt_index

            # -------------------------------------------------
            # TP
            # -------------------------------------------------

            if (
                best_gt_index >= 0
                and
                best_iou >= iou_threshold
            ):

                true_positive += 1

                matched_ground_truth.add(
                    best_gt_index
                )

            # -------------------------------------------------
            # FP
            # -------------------------------------------------

            else:

                false_positive += 1

        # -----------------------------------------------------
        # FN
        # -----------------------------------------------------

        false_negative = (
            len(ground_truth_boxes)
            -
            len(matched_ground_truth)
        )

        return (
            true_positive,
            false_positive,
            false_negative
        )

    # =========================================================
    # Precision
    # =========================================================

    def calculate_precision(
        self,
        true_positive,
        false_positive
    ):

        denominator = (
            true_positive +
            false_positive
        )

        if denominator == 0:

            return 0.0

        return (
            true_positive /
            denominator
        )

    # =========================================================
    # Recall
    # =========================================================

    def calculate_recall(
        self,
        true_positive,
        false_negative
    ):

        denominator = (
            true_positive +
            false_negative
        )

        if denominator == 0:

            return 0.0

        return (
            true_positive /
            denominator
        )

    # =========================================================
    # F1
    # =========================================================

    def calculate_f1(
        self,
        precision,
        recall
    ):

        denominator = (
            precision +
            recall
        )

        if denominator == 0:

            return 0.0

        return (
            2 *
            precision *
            recall /
            denominator
        )

    # =========================================================
    # Calculate AP for ONE CLASS
    #
    # Standard detection AP calculation
    # =========================================================

    def calculate_ap(
        self,
        predictions,
        ground_truths,
        class_id,
        iou_threshold
    ):

        # -----------------------------------------------------
        # Ground truth count for this class
        # -----------------------------------------------------

        total_ground_truth = 0

        for image_id in ground_truths:

            for gt in ground_truths[image_id]:

                if gt["label"] == class_id:

                    total_ground_truth += 1

        if total_ground_truth == 0:

            return 0.0

        # -----------------------------------------------------
        # Get predictions of this class
        # -----------------------------------------------------

        class_predictions = [

            prediction

            for prediction in predictions

            if prediction["label"] == class_id

        ]

        # -----------------------------------------------------
        # No predictions
        # -----------------------------------------------------

        if len(class_predictions) == 0:

            return 0.0

        # -----------------------------------------------------
        # Sort by confidence
        # -----------------------------------------------------

        class_predictions = sorted(

            class_predictions,

            key=lambda x: x["score"],

            reverse=True

        )

        true_positive = np.zeros(
            len(class_predictions)
        )

        false_positive = np.zeros(
            len(class_predictions)
        )

        # -----------------------------------------------------
        # Track matched GT
        #
        # Each image has its own GT
        # -----------------------------------------------------

        matched_ground_truth = {}

        for image_id in ground_truths:

            matched_ground_truth[image_id] = set()

        # -----------------------------------------------------
        # Match predictions
        # -----------------------------------------------------

        for prediction_index, prediction in enumerate(
            class_predictions
        ):

            image_id = prediction["image_id"]

            pred_box = prediction["box"]

            best_iou = 0.0

            best_gt_index = -1

            image_ground_truths = ground_truths.get(
                image_id,
                []
            )

            for gt_index, gt in enumerate(
                image_ground_truths
            ):

                if gt_index in matched_ground_truth[image_id]:
                    continue

                if gt["label"] != class_id:
                    continue

                iou = self.calculate_iou(
                    pred_box,
                    gt["box"]
                )

                if iou > best_iou:

                    best_iou = iou
                    best_gt_index = gt_index

            # -------------------------------------------------
            # TP
            # -------------------------------------------------

            if (
                best_gt_index >= 0
                and
                best_iou >= iou_threshold
            ):

                true_positive[prediction_index] = 1

                matched_ground_truth[
                    image_id
                ].add(
                    best_gt_index
                )

            # -------------------------------------------------
            # FP
            # -------------------------------------------------

            else:

                false_positive[prediction_index] = 1

        # -----------------------------------------------------
        # Cumulative values
        # -----------------------------------------------------

        cumulative_tp = np.cumsum(
            true_positive
        )

        cumulative_fp = np.cumsum(
            false_positive
        )

        precision = (
            cumulative_tp /
            np.maximum(
                cumulative_tp +
                cumulative_fp,
                1e-12
            )
        )

        recall = (
            cumulative_tp /
            total_ground_truth
        )

        # -----------------------------------------------------
        # Precision envelope
        # -----------------------------------------------------

        precision = np.concatenate(
            (
                [0.0],
                precision,
                [0.0]
            )
        )

        recall = np.concatenate(
            (
                [0.0],
                recall,
                [1.0]
            )
        )

        for index in range(
            len(precision) - 2,
            -1,
            -1
        ):

            precision[index] = max(
                precision[index],
                precision[index + 1]
            )

        # -----------------------------------------------------
        # Calculate AP
        # -----------------------------------------------------

        indices = np.where(
            recall[1:] != recall[:-1]
        )[0]

        ap = np.sum(
            (
                recall[indices + 1]
                -
                recall[indices]
            )
            *
            precision[indices + 1]
        )

        return float(ap)

    # =========================================================
    # Evaluate one model
    # =========================================================

    def evaluate(
        self,
        model,
        dataset,
        device,
        prediction_function
    ):

        model.eval()

        # =====================================================
        # Precision / Recall / F1
        # =====================================================

        total_true_positive = 0
        total_false_positive = 0
        total_false_negative = 0

        # =====================================================
        # Store ALL predictions
        #
        # Important:
        # NO confidence filtering for mAP
        # =====================================================

        all_predictions = []

        # =====================================================
        # Ground truths by image
        # =====================================================

        ground_truths = {}

        # =====================================================
        # Ground truth classes
        # =====================================================

        ground_truth_per_class = {}

        # =====================================================
        # Matched IoU
        # =====================================================

        matched_ious = []

        # =====================================================
        # Loop validation dataset
        # =====================================================

        for image_index in range(
            len(dataset)
        ):

            image, target = dataset[
                image_index
            ]

            ground_truth_boxes = (
                target["boxes"]
            )

            ground_truth_labels = (
                target["labels"]
            )

            # -------------------------------------------------
            # Store GT
            # -------------------------------------------------

            ground_truths[image_index] = []

            for gt_box, gt_label in zip(
                ground_truth_boxes,
                ground_truth_labels
            ):

                class_id = int(
                    gt_label.item()
                )

                ground_truths[
                    image_index
                ].append(
                    {
                        "box":
                            gt_box.cpu().numpy(),

                        "label":
                            class_id
                    }
                )

                ground_truth_per_class[
                    class_id
                ] = (
                    ground_truth_per_class.get(
                        class_id,
                        0
                    )
                    +
                    1
                )

            # -------------------------------------------------
            # Prediction
            # -------------------------------------------------

            with torch.no_grad():

                prediction = (
                    prediction_function(
                        model,
                        image,
                        device
                    )
                )

            prediction_boxes = (
                prediction["boxes"]
            )

            prediction_labels = (
                prediction["labels"]
            )

            prediction_scores = (
                prediction["scores"]
            )

            # =================================================
            # STORE ALL PREDICTIONS FOR mAP
            # =================================================

            for box, label, score in zip(
                prediction_boxes,
                prediction_labels,
                prediction_scores
            ):

                all_predictions.append(
                    {
                        "image_id":
                            image_index,

                        "box":
                            box.cpu().numpy(),

                        "label":
                            int(
                                label.item()
                            ),

                        "score":
                            float(
                                score.item()
                            )
                    }
                )

            # =================================================
            # Confidence filtering
            #
            # ONLY for:
            # Precision
            # Recall
            # F1
            # IoU
            # =================================================

            keep = (
                prediction_scores
                >=
                self.confidence_threshold
            )

            filtered_boxes = (
                prediction_boxes[keep]
            )

            filtered_labels = (
                prediction_labels[keep]
            )

            filtered_scores = (
                prediction_scores[keep]
            )

            # =================================================
            # Precision / Recall / F1
            # =================================================

            (
                true_positive,
                false_positive,
                false_negative
            ) = self.match_predictions(

                filtered_boxes,

                filtered_labels,

                filtered_scores,

                ground_truth_boxes,

                ground_truth_labels,

                self.iou_threshold

            )

            total_true_positive += (
                true_positive
            )

            total_false_positive += (
                false_positive
            )

            total_false_negative += (
                false_negative
            )

            # =================================================
            # IoU
            # =================================================

            used_ground_truth = set()

            if len(filtered_scores) > 0:

                order = torch.argsort(
                    filtered_scores,
                    descending=True
                )

                for prediction_index in order:

                    pred_box = (
                        filtered_boxes[
                            prediction_index
                        ]
                    )

                    pred_label = int(
                        filtered_labels[
                            prediction_index
                        ].item()
                    )

                    best_iou = 0.0
                    best_gt_index = -1

                    for gt_index, (
                        gt_box,
                        gt_label
                    ) in enumerate(
                        zip(
                            ground_truth_boxes,
                            ground_truth_labels
                        )
                    ):

                        if gt_index in used_ground_truth:
                            continue

                        if (
                            pred_label
                            !=
                            int(
                                gt_label.item()
                            )
                        ):
                            continue

                        iou = self.calculate_iou(

                            pred_box.cpu().numpy(),

                            gt_box.cpu().numpy()

                        )

                        if iou > best_iou:

                            best_iou = iou
                            best_gt_index = gt_index

                    if (
                        best_gt_index >= 0
                        and
                        best_iou >= self.iou_threshold
                    ):

                        matched_ious.append(
                            best_iou
                        )

                        used_ground_truth.add(
                            best_gt_index
                        )

        # =====================================================
        # Precision
        # =====================================================

        precision = (
            self.calculate_precision(
                total_true_positive,
                total_false_positive
            )
        )

        # =====================================================
        # Recall
        # =====================================================

        recall = (
            self.calculate_recall(
                total_true_positive,
                total_false_negative
            )
        )

        # =====================================================
        # F1
        # =====================================================

        f1 = (
            self.calculate_f1(
                precision,
                recall
            )
        )

        # =====================================================
        # Mean IoU
        # =====================================================

        if len(matched_ious) > 0:

            mean_iou = float(
                np.mean(
                    matched_ious
                )
            )

        else:

            mean_iou = 0.0

        # =====================================================
        # Classes
        # =====================================================

        class_ids = range(1, 7)

        # =====================================================
        # mAP@0.5
        # =====================================================

        ap50_values = []

        for class_id in class_ids:

            ap50 = self.calculate_ap(

                all_predictions,

                ground_truths,

                class_id,

                0.50

            )

            ap50_values.append(
                ap50
            )

        if len(ap50_values) > 0:

            map50 = float(
                np.mean(
                    ap50_values
                )
            )

        else:

            map50 = 0.0

        # =====================================================
        # mAP@0.5:0.95
        #
        # IoU:
        # 0.50
        # 0.55
        # 0.60
        # ...
        # 0.95
        # =====================================================

        map_values = []

        for iou_threshold in (
            self.map_iou_thresholds
        ):

            ap_values = []

            for class_id in class_ids:

                ap = self.calculate_ap(

                    all_predictions,

                    ground_truths,

                    class_id,

                    float(iou_threshold)

                )

                ap_values.append(
                    ap
                )

            if len(ap_values) > 0:

                map_at_threshold = float(
                    np.mean(
                        ap_values
                    )
                )

            else:

                map_at_threshold = 0.0

            map_values.append(
                map_at_threshold
            )

        if len(map_values) > 0:

            map5095 = float(
                np.mean(
                    map_values
                )
            )

        else:

            map5095 = 0.0

        # =====================================================
        # Result
        # =====================================================

        return {

            "Precision":
                precision,

            "Recall":
                recall,

            "F1-score":
                f1,

            "IoU":
                mean_iou,

            "mAP@0.5":
                map50,

            "mAP@0.5:0.95":
                map5095,

            "TP":
                total_true_positive,

            "FP":
                total_false_positive,

            "FN":
                total_false_negative,

            "Ground Truth":
                sum(
                    ground_truth_per_class.values()
                )

        }