"""
Definitions of the employed loss function and metrics.
"""


from tkinter import Y
from typing import List, Tuple

from numpy import arange, ndarray
# pylint: disable=import-error
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
# pylint: enable=import-error

if __name__ != 'main_by_mattia':
    from common_constants import (
        DATA_TYPE_FOR_OUTPUTS,
        LOSS_CONTRIBUTE_IMPORTANCE_OF_EMPTY_ANCHORS,
        LOSS_CONTRIBUTE_IMPORTANCE_OF_FULL_ANCHORS,
        OUTPUT_GRID_N_ROWS,
        OUTPUT_GRID_N_COLUMNS,
        N_ANCHORS_PER_CELL,
        N_OUTPUTS_PER_ANCHOR
    )
    from inference import (
        get_bounding_boxes_from_model_outputs
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
OBJECTNESS_PROBABILITY_THRESHOLD = 0.5  # FIXME: not required


def evaluate_bounding_boxes_matching(
        expected_bounding_boxes: Tensor,
        predicted_bounding_boxes: Tensor,
        iou_threshold: float
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    TODO
    """
    return (
        false_positives,
        false_negatives,
        true_positives
    )


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
    labels_as_bounding_boxes = get_bounding_boxes_from_model_outputs(
        model_outputs=y_true,
        from_labels=True
    )

    mean_f2_scores_for_different_iou_thresholds = []

    for threshold in IOU_THRESHOLDS:
        (
            false_positives,
            false_negatives,
            true_positives
        ) = evaluate_bounding_boxes_matching(
            expected_bounding_boxes=labels_as_bounding_boxes,
            predicted_bounding_boxes=y_pred,
            iou_threshold=threshold
        )

        mean_f2_scores_for_different_iou_thresholds.append(
            # ----------------------------------------------------------------
            # convert_to_tensor(
            #     value=[
            #         mean_f2_scores(
            #             false_positives=false_positives,
            #             false_negatives=false_negatives,
            #             true_positives=true_positives
            #         )
            #     ],
            #     dtype=DATA_TYPE_FOR_OUTPUTS
            # )
            # ----------------------------------------------------------------
            mean_f2_scores(
                false_positives=false_positives,
                false_negatives=false_negatives,
                true_positives=true_positives
            )
        )

    return reduce_mean(
        input_tensor=stack(
            values=mean_f2_scores_for_different_iou_thresholds,
            axis=-1
        ),
        axis=-1
    )


def mean_f2_scores(
        false_positives: Tensor,
        false_negatives: Tensor,
        true_positives: Tensor
) -> Tensor:
    """
    Return the F2-scores of each mini-batch sample, given their numbers of
    false positives, false negatives and true positives as inputs.
    """
    # FIXME: vectorize considering batches and with TF
    return (
        true_positives /
        (true_positives + 0.8*false_negatives + 0.2*false_positives + EPSILON)
    )


def yolov3_variant_loss(y_true: Tensor, y_pred: Tensor) -> Tensor:
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
                y=OBJECTNESS_PROBABILITY_THRESHOLD
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
