"""
Utilities for inference time, for converting model outputs to bounding boxes' predictions.
"""


from typing import List, Tuple

# pylint: disable=import-error
from tensorflow import (
    concat,
    convert_to_tensor,
    expand_dims,
    reshape,
    Tensor,
    tile
)
from tensorflow.image import combined_non_max_suppression
# pylint: enable=import-error

if __name__ != 'main_by_mattia':
    from common_constants import (
        N_OUTPUTS_PER_ANCHOR,
        OUTPUT_GRID_CELL_CORNERS_XY_COORDS,
        OUTPUT_GRID_CELL_N_COLUMNS,
        OUTPUT_GRID_CELL_N_ROWS
    )
    from model_architecture import YOLOv3Variant
    from samples_and_labels import (
        dataset_of_samples_and_model_outputs,
        split_dataset_into_batched_training_and_validation_sets
    )


IOU_THRESHOLD_FOR_NON_MAXIMUM_SUPPRESSION = 0.5
MINIMUM_BOUNDING_BOX_SIDE_DIMENSION_TOLERANCE = 0.1
MAXIMUM_N_BOUNDING_BOXES_AFTER_NMS = 100
SCORE_THRESHOLD_FOR_NON_MAXIMUM_SUPPRESSION = 0.5


def batched_anchors_rel_to_abs_x_y_w_h(
        batched_anchors_relative_x_y_w_h: Tensor,
        batched_anchors_corners_absolute_x_y: Tensor
) -> Tensor:
    """
    TODO
    """
    pass


def batched_anchors_x_y_w_h_to_x1_y1_x2_y2(
        batched_anchors_absolute_x_y_w_h: Tensor
) -> Tensor:
    """
    TODO
    """
    pass


def batched_anchors_x1_y1_x2_y2_to_x_y_w_h(
        batched_anchors_absolute_x1_y1_x2_y2: Tensor
) -> Tensor:
    """
    TODO
    """
    pass


def convert_bounding_boxes_to_submission_format(
        bounding_boxes: Tensor
) -> str:
    """
    TODO
    """
    raise NotImplementedError


def get_bounding_boxes_from_model_outputs(
        model_outputs: Tensor,
        from_labels: bool = False
) -> Tensor:
    """
    format conversion
    post-processing, non-maximum suppression
    un-normalization, reconstruction
    TODO
    NOTE: in my approach, anchors are just used to create labels as relative
    aspect ratios, neither to recreate predictions nor as absolute sizes -
    that's why anchors are not used here
    ---
        Input Shape:
            - (
                VARIABLE_N_SAMPLES,
                OUTPUT_GRID_N_ROWS,
                OUTPUT_GRID_N_COLUMNS,
                N_ANCHORS_PER_CELL,
                N_OUTPUTS_PER_ANCHOR
            )
    """
    n_mini_batch_samples = model_outputs.shape[0]

    # turning the model outputs into flattened (except along the batch
    # dimension) anchor predictions:
    anchors_outputs = reshape(
        tensor=model_outputs,
        shape=(n_mini_batch_samples, -1, N_OUTPUTS_PER_ANCHOR)
    )  # shape → (samples, anchors_per_image, attributes)

    # crating a corresponding tensor of flattened (except along the batch
    # dimension) anchor corners' absolute (x, y) coordinates - NOTE:
    # TensorFlow uses a row-major ordering for reshaping, but the following
    # procedure ensures that the same ordering as for flattened anchor outputs
    # is followed:
    anchors_corners_absolute_x_y = reshape(
        tensor=tile(
            input=expand_dims(
                input=tile(
                    input=expand_dims(
                        input=convert_to_tensor(
                            value=OUTPUT_GRID_CELL_CORNERS_XY_COORDS
                        ),
                        # shape → (rows, columns, 2)
                        axis=0
                    ),  # shape → (1, rows, columns, 2)
                    multiples=(n_mini_batch_samples, 1, 1, 1)
                ),  # shape → (samples, rows, columns, 2)
                axis=3
            ),  # shape → (samples, rows, columns, 1, 2)
            multiples=(n_mini_batch_samples, 1, 1, 1)
        ),  # shape → (samples, rows, columns, anchors_per_cell, 2)
        shape=(n_mini_batch_samples, -1, 2)
    )  # shape → (samples, anchors_per_image, 2)

    # when the model outputs are intended as labels:
    if from_labels:
        # non-maximum suppression and the IoU threshold are not relevant when
        # the model outputs represent labels as they are already discretized:
        bounding_boxes_scores_plus_absolute_x_y_w_h = None  # TODO
        # (Given a value a)
        # To achieve this in numpy you just have to write :
        # selected_rows = myarray[myarray[:,0]== a]
        # In tensorflow, use tf.where :
        # mytensor[tf.squeeze(tf.where(tf.equal(mytensor[:,0],a), None, None))
        raise NotImplementedError

    # when the model outputs are intended as predictions:
    else:
        # applying non-maximum suppression to generate robust bounding box
        # candidates with respective reliability scores:

        # adding a dummy class dimension for the later Tensorflow's function
        # application - NOTE: a single class in considered in the task of
        # interest:
        anchors_outputs = expand_dims(input=anchors_outputs, axis=2)
        # shape → (samples, anchors_per_image, 1, attributes)

        anchors_scores = anchors_outputs[..., 0]
        # shape → (samples, anchors_per_image, 1)

        anchors_relative_x_y_w_h = anchors_outputs[..., 1:]
        # shape → (samples, anchors_per_image, 1, 4)

        anchors_absolute_x_y_w_h = batched_anchors_rel_to_abs_x_y_w_h(
            batched_anchors_relative_x_y_w_h=anchors_relative_x_y_w_h,
            batched_anchors_corners_absolute_x_y=anchors_corners_absolute_x_y
        )  # shape → (samples, anchors_per_image, 1, 4)

        anchors_absolute_x1_y1_x2_y2 = batched_anchors_x_y_w_h_to_x1_y1_x2_y2(
            batched_anchors_absolute_x_y_w_h=anchors_absolute_x_y_w_h
        )  # shape → (samples, anchors_per_image, 1, 4)

        (
            boxes_absolute_x1_y1_x2_y2,  # shape → (samples, boxes, 4)
            boxes_scores,  # shape → (samples, boxes)
            _,  # class for each sample
            _  # number of detections for each sample
        ) = combined_non_max_suppression(
            boxes=anchors_absolute_x1_y1_x2_y2,
            scores=anchors_scores,
            max_output_size_per_class=MAXIMUM_N_BOUNDING_BOXES_AFTER_NMS,
            # NOTE: a single class in considered in the task of interest:
            max_total_size=MAXIMUM_N_BOUNDING_BOXES_AFTER_NMS,
            iou_threshold=IOU_THRESHOLD_FOR_NON_MAXIMUM_SUPPRESSION,
            score_threshold=SCORE_THRESHOLD_FOR_NON_MAXIMUM_SUPPRESSION,
            pad_per_class=False,
            clip_boxes=True
        )

        boxes_absolute_x_y_w_h = batched_anchors_x1_y1_x2_y2_to_x_y_w_h(
            batched_anchors_absolute_x1_y1_x2_y2=boxes_absolute_x1_y1_x2_y2
        )  # shape → (samples, boxes, 4)

        bounding_boxes_scores_plus_absolute_x_y_w_h = concat(
            values=(
                expand_dims(input=boxes_scores, axis=-1),
                # shape → (samples, boxes, 1)
                boxes_absolute_x_y_w_h
                # shape → (samples, boxes, 4)
            ),
            axis = -1
        )  # shape → (samples, boxes, 5)
        # --------------------------------------------------------------------
        # post_processed_bounding_boxes = generate_bounding_box_proposals(
        #     scores=model_outputs[..., 0],
        #     bbox_deltas=model_outputs[..., 1:],  # TODO
        #     # image_info=, TODO
        #     # NOTE: in my approach, anchors are just used to create labels as
        #     # relative aspect ratios, neither to recreate predictions nor as
        #     # absolute sizes - that's why here anchors are passed all with dummy
        #     # values representing unitary multiplication factors that produce no
        #     # changes in size:
        #     anchors=([[1] * N_OUTPUTS_PER_ANCHOR] * N_ANCHORS_PER_CELL),
        #     nms_threshold=0.7,
        #     # all anchor's outputs are considered for NMS:
        #     pre_nms_topn=N_ANCHORS_PER_IMAGE,
        #     # NOTE: minimum bounding box size set according to dataset inspection
        #     # results - adding a tolerance threshold as an assumption that
        #     # slightly smaller bounding boxes can be observed:
        #     min_size=int(
        #         min(MINIMUM_BOUNDING_BOX_HEIGHT, MINIMUM_BOUNDING_BOX_WIDTH) *
        #         MINIMUM_BOUNDING_BOX_SIDE_DIMENSION_TOLERANCE
        #     ),
        #     post_nms_topn=MAXIMUM_N_BOUNDING_BOXES_AFTER_NMS
        # )
        # --------------------------------------------------------------------

    raise bounding_boxes_scores_plus_absolute_x_y_w_h


if __name__ == '__main__':
    (
        training_samples_and_labels, validation_samples_and_labels
    ) = split_dataset_into_batched_training_and_validation_sets(
        training_plus_validation_set=dataset_of_samples_and_model_outputs()
    )

    model = YOLOv3Variant()

    for samples, labels in training_samples_and_labels:
        _ = get_bounding_boxes_from_model_outputs(
            model_outputs=labels,
            from_labels=True
        )
        predictions = model(samples)
        _ = get_bounding_boxes_from_model_outputs(
            model_outputs=predictions,
            from_labels=False
        )
        break
