"""
Definitions of the employed loss function and metrics.

NOTE on the employed metric, citing the competition explanation:
------------------------------------------------------------------------------
"This competition is evaluated on the F2 Score at different intersection over
union (IoU) thresholds. The F2 metric weights recall more heavily than
precision, as in this case it makes sense to tolerate some false positives
in order to ensure very few starfish are missed.

The metric sweeps over IoU thresholds in the range of 0.3 to 0.8 with a step
size of 0.05, calculating an F2 score at each threshold. For example, at a
threshold of 0.5, a predicted object is considered a "hit" if its IoU with a
ground truth object is at least 0.5.

A true positive is the first (in confidence order, see details below)
submission box in a sample with an IoU greater than the threshold against an
unmatched solution box.

Once all submission boxes have been evaluated, any unmatched submission boxes
are false positives; any unmatched solution boxes are false negatives.

The final F2 Score is calculated as the mean of the F2 scores at each IoU
threshold. Within each IoU threshold the competition metric uses micro
averaging; every true positive, false positive, and false negative has equal
weight compared to each other true positive, false positive, and false
negative.

In your submission, you are also asked to provide a confidence level for each
bounding box. Bounding boxes are evaluated in order of their confidence
levels. This means that bounding boxes with higher confidence will be checked
first for matches against solutions, which determines what boxes are
considered true and false positives."
------------------------------------------------------------------------------
"""


from typing import List, Tuple

# pylint: disable=import-error,no-name-in-module
from tensorflow import (
    expand_dims,
    stack,
    Tensor,
    tile,
    where,
    zeros
)
from tensorflow.keras.losses import binary_crossentropy, mean_absolute_error
from tensorflow.math import (
    add,
    greater_equal,
    logical_not,
    multiply,
    reduce_mean,
    reduce_sum
)
# pylint: enable=import-error,no-name-in-module

# only when running everything in a unified notebook on Kaggle's servers:
if __name__ != 'main_by_mattia':
    from common_constants import (
        LOSS_CONTRIBUTE_IMPORTANCE_OF_EMPTY_ANCHORS,
        LOSS_CONTRIBUTE_IMPORTANCE_OF_FULL_ANCHORS,
        OUTPUT_GRID_N_ROWS,
        OUTPUT_GRID_N_COLUMNS,
        N_ANCHORS_PER_CELL,
        N_OUTPUTS_PER_ANCHOR
    )
    from inference import (
        get_bounding_boxes_from_model_outputs,
        convert_batched_bounding_boxes_to_final_format
    )
    from samples_and_labels import (
        MINI_BATCH_SIZE
    )


EPSILON = 1e-7
IOU_THRESHOLDS = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
LABELS_FULL_SHAPE = (
    MINI_BATCH_SIZE,
    OUTPUT_GRID_N_ROWS,
    OUTPUT_GRID_N_COLUMNS,
    N_ANCHORS_PER_CELL,
    N_OUTPUTS_PER_ANCHOR
)


def compute_intersection_over_union(
        x_y_w_h_first_box: Tuple[int, int, int, int],
        x_y_w_h_second_box: Tuple[int, int, int, int]
) -> float:
    """
    Compute the intersection over union (IoU) between two boxes represented
    by the two integer sets of {top-left corner x coordinate, top-left corner
    y coordinate, box width, box height} given as inputs.
    """
    # boxes intersection area:
    boxes_intersection_area = (
        (  # x-side intersection length:
            max(
                (
                    min(
                        x_y_w_h_first_box[0] + x_y_w_h_first_box[2],
                        x_y_w_h_second_box[0] + x_y_w_h_second_box[2]
                    ) - max(
                        x_y_w_h_first_box[0],
                        x_y_w_h_second_box[0]
                    )
                ),
                0
            )
        ) * (  # y-side intersection length:
            max(
                (
                    min(
                        x_y_w_h_first_box[1] + x_y_w_h_first_box[3],
                        x_y_w_h_second_box[1] + x_y_w_h_second_box[3]
                    ) - max(
                        x_y_w_h_first_box[1],
                        x_y_w_h_second_box[1]
                    )
                ),
                0
            )
        )
    )

    return (
        boxes_intersection_area / (  # boxes union area:
            (x_y_w_h_first_box[2] * x_y_w_h_first_box[3])  # 1st box area:
            + (x_y_w_h_second_box[2] * x_y_w_h_second_box[3])  # 2nd box area
            - boxes_intersection_area
        )
    )


def compute_mean_f2_scores(
        images_matches: List[Tuple[float, float, float]]
) -> float:
    """
    Return the F2-scores of each mini-batch sample, given their numbers of
    false positives, false negatives and true positives as inputs.
    """
    cumulative_f2_score = 0
    number_of_f2_scores_summed = 0
    for true_positives, false_positives, false_negatives in images_matches:
        number_of_f2_scores_summed += 1
        cumulative_f2_score += (
            true_positives / (
                true_positives + 0.8*false_negatives + 0.2*false_positives
                + EPSILON
            )
        )
    return cumulative_f2_score / number_of_f2_scores_summed


def evaluate_batched_bounding_boxes_matching(
        expected_bounding_boxes: List[Tuple[float, int, int, int, int]],
        predicted_bounding_boxes: List[Tuple[float, int, int, int, int]],
        iou_threshold: float
) -> List[Tuple[int, int, int]]:
    """
    Retun the true positives, false positives, false negatives - according to
    the competition metric definition - for each pair of arrays of predicted
    vs expected bounding boxes in the batched inputs.
    """
    matches = []
    for image_expected_bounding_boxes, image_predicted_bounding_boxes in zip(
            expected_bounding_boxes, predicted_bounding_boxes
    ):
        # sorting the predicted bounding boxes of the considered image by
        # relevance according to the predicted confidence score:
        best_to_worst_predicted_bounding_boxes = sorted(
            image_predicted_bounding_boxes,
            key=lambda bounding_box_attributes: bounding_box_attributes[0],
            reverse=True
        )
        # NOTE: the expected bounding boxes are equally important, no sorting
        # is required

        true_positives = 0
        false_positives = 0

        for predicted_bounding_box in best_to_worst_predicted_bounding_boxes:
            current_bounding_box_matched = True

            for index, expected_bounding_box in enumerate(
                    image_expected_bounding_boxes
            ):
                if (
                        compute_intersection_over_union(
                            x_y_w_h_first_box=predicted_bounding_box[1:],
                            x_y_w_h_second_box=expected_bounding_box[1:]
                        ) >= iou_threshold
                ):
                    current_bounding_box_matched = True
                    del image_expected_bounding_boxes[index]
                    true_positives += 1
                    break

            if not current_bounding_box_matched:
                false_positives += 1

        false_negatives = len(image_expected_bounding_boxes)

        matches.append([true_positives, false_positives, false_negatives])

    return matches


def iou_threshold_averaged_f2_score(y_true: Tensor, y_pred: Tensor) -> Tensor:
    """
    Metric used to validate the model goodness - according to the competition
    aim - that represents the F2 score, as they decided to favor recall twice
    as much as precision, avereaged over different IoU thresholds for
    considering bounding boxes as detected or not, with these thresholds
    being: {0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8}.
    """
    # turning the labels representing model outputs into bounding boxes,
    # following the same format that the predictions assume at inference time,
    # when they undergo an additional post-processing, unlike during training:
    (
        labels_bounding_boxes, labels_n_valid_bounding_boxes
    ) = get_bounding_boxes_from_model_outputs(
        model_outputs=y_true,
        from_labels=True
    )
    (
        predictions_bounding_boxes, predictions_n_valid_bounding_boxes
    ) = get_bounding_boxes_from_model_outputs(
        model_outputs=y_pred,
        from_labels=False
    )

    labels_as_lists_of_bounding_boxes = (
        convert_batched_bounding_boxes_to_final_format(
            batched_bounding_boxes=labels_bounding_boxes,
            batched_n_valid_bounding_boxes=labels_n_valid_bounding_boxes,
            predicting_online=False,
            as_strings=False
        )
    )
    predictions_as_lists_of_bounding_boxes = (
        convert_batched_bounding_boxes_to_final_format(
            batched_bounding_boxes=predictions_bounding_boxes,
            batched_n_valid_bounding_boxes=predictions_n_valid_bounding_boxes,
            predicting_online=False,
            as_strings=False
        )
    )

    mean_f2_scores_for_different_iou_thresholds = []

    for threshold in IOU_THRESHOLDS:
        mean_f2_scores_for_different_iou_thresholds.append(
            compute_mean_f2_scores(
                images_matches=evaluate_batched_bounding_boxes_matching(
                    expected_bounding_boxes=labels_as_lists_of_bounding_boxes,
                    predicted_bounding_boxes=(
                        predictions_as_lists_of_bounding_boxes
                    ),
                    iou_threshold=threshold
                )
            )
        )

    return reduce_mean(
        input_tensor=stack(
            values=mean_f2_scores_for_different_iou_thresholds,
            axis=-1
        ),
        axis=-1
    )


def yolov3_variant_loss(y_true: Tensor, y_pred: Tensor) -> Tensor:  # noqa: E501 pylint: disable=too-many-locals
    """
    Loss function minimized to train the defined YOLOv3 variant.
    ---
        Input Shapes:
            - (
                MINI_BATCH_SIZE,
                OUTPUT_GRID_N_ROWS,
                OUTPUT_GRID_N_COLUMNS,
                N_ANCHORS_PER_CELL,
                N_OUTPUTS_PER_ANCHOR
            )
            - (
                MINI_BATCH_SIZE,
                OUTPUT_GRID_N_ROWS,
                OUTPUT_GRID_N_COLUMNS,
                N_ANCHORS_PER_CELL,
                N_OUTPUTS_PER_ANCHOR
            )
    ---
        Output Shape:
            - (MINI_BATCH_SIZE,)
    """
    dummy_zeros_to_get_no_loss = zeros(shape=LABELS_FULL_SHAPE)

    true_anchors_with_objects_flags = tile(
        input=expand_dims(
            input=greater_equal(
                x=y_true[..., 0],
                y=0.5
            ),
            axis=-1
        ),
        multiples=(1, 1, 1, 1, N_OUTPUTS_PER_ANCHOR)
    )  # shape → (samples, rows, columns, anchors, attributes)
    true_anchors_without_objects_flags = logical_not(
        x=true_anchors_with_objects_flags
    )  # shape → (samples, rows, columns, anchors, attributes)

    y_true_full_anchors = expand_dims(
        input=where(
            condition=true_anchors_with_objects_flags,
            x=y_true,
            y=dummy_zeros_to_get_no_loss
        ),
        axis=-1
    )  # shape → (samples, rows, columns, anchors, attributes, 1)
    y_true_empty_anchors = expand_dims(
        input=where(
            condition=true_anchors_without_objects_flags,
            x=y_true,
            y=dummy_zeros_to_get_no_loss
        ),
        axis=-1
    )  # shape → (samples, rows, columns, anchors, attributes, 1)
    y_pred_full_anchors = expand_dims(
        input=where(
            condition=true_anchors_with_objects_flags,
            x=y_pred,
            y=dummy_zeros_to_get_no_loss
        ),
        axis=-1
    )  # shape → (samples, rows, columns, anchors, attributes, 1)
    y_pred_empty_anchors = expand_dims(
        input=where(
            condition=true_anchors_without_objects_flags,
            x=y_pred,
            y=dummy_zeros_to_get_no_loss
        ),
        axis=-1
    )  # shape → (samples, rows, columns, anchors, attributes, 1)

    full_anchors_objectness_loss_per_anchor = binary_crossentropy(
        y_true=y_true_full_anchors[..., 0, :],
        y_pred=y_pred_full_anchors[..., 0, :],
        from_logits=False,
        axis=-1,
    )  # shape → (samples, rows, columns, anchors)

    empty_anchors_objectness_loss_per_anchor = binary_crossentropy(
        y_true=y_true_empty_anchors[..., 0, :],
        y_pred=y_pred_empty_anchors[..., 0, :],
        from_logits=False,
        axis=-1,
    )  # shape → (samples, rows, columns, anchors)

    full_anchors_coordinates_offsets_loss_per_anchor = reduce_sum(
        input_tensor=mean_absolute_error(
            y_true=y_true_full_anchors[..., 1:3, :],
            y_pred=y_pred_full_anchors[..., 1:3, :],
        ),
        axis=-1
    )  # shape → (samples, rows, columns, anchors)
    full_anchors_coordinates_scales_loss_per_anchor = reduce_sum(
        input_tensor=mean_absolute_error(
            y_true=y_true_full_anchors[..., 3:, :],
            y_pred=y_pred_full_anchors[..., 3:, :],
        ),
        axis=-1
    )  # shape → (samples, rows, columns, anchors)

    full_anchors_coordinates_loss_per_anchor = add(
        x=full_anchors_coordinates_offsets_loss_per_anchor,
        y=full_anchors_coordinates_scales_loss_per_anchor
    )  # shape → (samples, rows, columns, anchors)

    full_anchors_mean_loss = reduce_mean(
        input_tensor=add(
            x=full_anchors_objectness_loss_per_anchor,
            y=full_anchors_coordinates_loss_per_anchor
        ),
        axis=[1, 2, 3]
    )  # shape → (samples,)

    empty_anchors_mean_loss = reduce_mean(
        input_tensor=empty_anchors_objectness_loss_per_anchor,
        axis=[1, 2, 3]
    )  # shape → (samples,)

    # NOTE: without weighting, here, after mean reduction, it means that both
    # terms will have the same weight, irrespectively of their imbalance
    return add(
        x=multiply(
            x=full_anchors_mean_loss,
            y=LOSS_CONTRIBUTE_IMPORTANCE_OF_FULL_ANCHORS
        ),
        y=multiply(
            x=empty_anchors_mean_loss,
            y=LOSS_CONTRIBUTE_IMPORTANCE_OF_EMPTY_ANCHORS
        )
    )  # shape → (samples,)


if __name__ == '__main__':
    raise NotImplementedError
