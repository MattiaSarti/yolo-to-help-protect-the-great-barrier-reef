"""
Utilities for inference time, for converting model outputs to bounding boxes' predictions.
"""


from typing import List, Tuple, Union

# pylint: disable=import-error
from tensorflow import (
    cast,
    clip_by_value,
    concat,
    convert_to_tensor,
    expand_dims,
    reshape,
    squeeze,
    stack,
    Tensor,
    tile
)
from tensorflow.image import combined_non_max_suppression
from tensorflow.math import (
    add,
    divide,
    subtract,
    multiply
)
# pylint: enable=import-error

if __name__ != 'main_by_mattia':
    from common_constants import (
        DATA_TYPE_FOR_OUTPUTS,
        IMAGE_N_COLUMNS,
        IMAGE_N_ROWS,
        N_ANCHORS_PER_CELL,
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


OBJECTNESS_PROBABILITY_THRESHOLD = 0.5  # FIXME: currently not required
IOU_THRESHOLD_FOR_NON_MAXIMUM_SUPPRESSION = 0.5
MINIMUM_BOUNDING_BOX_SIDE_DIMENSION_TOLERANCE = 0.1
MAXIMUM_N_BOUNDING_BOXES_AFTER_NMS = 100
SCORE_THRESHOLD_FOR_NON_MAXIMUM_SUPPRESSION = 0.5


def batched_anchors_rel_to_real_abs_x_y_w_h(
        batched_anchors_relative_x_y_w_h: Tensor,
        batched_anchors_corners_absolute_x_y: Tensor
) -> Tensor:
    """
    Turn batches of arrays of anchors where every anchor is represented by
    relative (x center, y center, w, h) values into batches of the same
    anchors where each anchor is represented by absolute (x top-left corner,
    y top-left corner, w, h) values - x and y represent respectively the x and
    y coordinates of the center in the inputs and of the top-left corner in
    the output, w and y represent respectively the width and height of sides.
    NOTE: this function changes not only the scale but also the meaning of x
    and y
    ---
        Input Shapes:
            - (
                VARIABLE_N_SAMPLES,
                N_ANCHORS_PER_IMAGE,
                1,
                4
            )
            - (
                VARIABLE_N_SAMPLES,
                N_ANCHORS_PER_IMAGE,
                2
            )
    ---
        Output Shape:
            - (
                VARIABLE_N_SAMPLES,
                N_ANCHORS_PER_IMAGE,
                1,
                4
            )
    """
    expanded_batched_anchors_corners_absolute_x_y = cast(
        # NOTE: they are already discretized, so any truncation due to casting
        # is not relevant
        x=expand_dims(
            input=batched_anchors_corners_absolute_x_y,
            axis=2
        ),
        dtype=DATA_TYPE_FOR_OUTPUTS
    )  # shape → (samples, anchors_per_image, 1, 2)

    batched_anchors_absolute_w = multiply(
        x=batched_anchors_relative_x_y_w_h[..., 2],
        y=IMAGE_N_COLUMNS
    )  # shape → (samples, anchors_per_image, 1)

    batched_anchors_absolute_h = multiply(
        x=batched_anchors_relative_x_y_w_h[..., 3],
        y=IMAGE_N_ROWS
    )  # shape → (samples, anchors_per_image, 1)

    batched_anchors_absolute_x = subtract(
        x=add(
            x=multiply(
                x=batched_anchors_relative_x_y_w_h[..., 0],
                y=float(OUTPUT_GRID_CELL_N_COLUMNS)
            ),  # shape → (samples, anchors_per_image, 1)
            y=expanded_batched_anchors_corners_absolute_x_y[..., 0]
        ),  # shape → (samples, anchors_per_image, 1)
        y=divide(
            x=batched_anchors_absolute_w,
            y=float(2)
        ),  # shape → (samples, anchors_per_image, 1)
    )  # shape → (samples, anchors_per_image, 1)

    batched_anchors_absolute_y = subtract(
        x=add(
            x=multiply(
                x=batched_anchors_relative_x_y_w_h[..., 1],
                y=float(OUTPUT_GRID_CELL_N_ROWS)
            ),  # shape → (samples, anchors_per_image, 1)
            y=expanded_batched_anchors_corners_absolute_x_y[..., 1]
        ),  # shape → (samples, anchors_per_image, 1)
        y=divide(
            x=batched_anchors_absolute_h,
            y=float(2)
        ),  # shape → (samples, anchors_per_image, 1)
    )  # shape → (samples, anchors_per_image, 1)

    return expand_dims(
        input=concat(
            values=(
                batched_anchors_absolute_x,
                batched_anchors_absolute_y,
                batched_anchors_absolute_w,
                batched_anchors_absolute_h
            ),
            axis=-1
        ),  # shape → (samples, anchors_per_image, 4)
        axis=2
    )  # shape → (samples, anchors_per_image, 1, 4)


def batched_anchors_x_y_w_h_to_x1_y1_x2_y2(
        batched_anchors_absolute_x_y_w_h: Tensor
) -> Tensor:
    """
    Turn batches of several anchors each where every anchor is represented by
    absolute (x, y, w, h) values, into batches of the same anchors where each
    anchor is represented by absolute (x1, y1, x2, y2) values - x and y
    represent respectively the x and y coordinates of the top-left corner, w
    and y represent respectively the width and height of sides, x1 and y1
    represent respectively the x and y coordinates of the top-left corner, x2
    and y2 represent respectively the x and y coordinates of the bottom-right
    corner - eventually clipping all output coordinates' values to fall inside
    the image.
    ---
        Input Shape:
            - (
                VARIABLE_N_SAMPLES,
                N_ANCHORS_PER_IMAGE,
                1,
                4
            )
    ---
        Output Shape:
            - (
                VARIABLE_N_SAMPLES,
                N_ANCHORS_PER_IMAGE,
                1,
                4
            )
    """
    batched_anchors_absolute_x1 = clip_by_value(
        t=batched_anchors_absolute_x_y_w_h[..., 0],
        # shape → (samples, anchors_per_image, 1)
        clip_value_min=0,
        clip_value_max=(IMAGE_N_COLUMNS - 1)
    )  # shape → (samples, anchors_per_image, 1)

    batched_anchors_absolute_y1 = clip_by_value(
        t=batched_anchors_absolute_x_y_w_h[..., 1],
        # shape → (samples, anchors_per_image, 1)
        clip_value_min=0,
        clip_value_max=(IMAGE_N_ROWS - 1)
    )  # shape → (samples, anchors_per_image, 1)

    batched_anchors_absolute_x2 = clip_by_value(
        t=add(
            x=batched_anchors_absolute_x_y_w_h[..., 0],
            y=batched_anchors_absolute_x_y_w_h[..., 2]
        ),  # shape → (samples, anchors_per_image, 1)
        clip_value_min=0,
        clip_value_max=(IMAGE_N_COLUMNS - 1)
    )  # shape → (samples, anchors_per_image, 1)

    batched_anchors_absolute_y2 = clip_by_value(
        t=add(
            x=batched_anchors_absolute_x_y_w_h[..., 1],
            y=batched_anchors_absolute_x_y_w_h[..., 3]
        ),  # shape → (samples, anchors_per_image, 1)
        clip_value_min=0,
        clip_value_max=(IMAGE_N_ROWS - 1)
    )  # shape → (samples, anchors_per_image, 1)

    return expand_dims(
        input=concat(
            values=(
                batched_anchors_absolute_x1,
                batched_anchors_absolute_y1,
                batched_anchors_absolute_x2,
                batched_anchors_absolute_y2
            ),
            axis=-1
        ),  # shape → (samples, anchors_per_image, 4)
        axis=2
    )  # shape → (samples, anchors_per_image, 1, 4)


def batched_anchors_x1_y1_x2_y2_to_x_y_w_h(
        batched_anchors_absolute_x1_y1_x2_y2: Tensor
) -> Tensor:
    """
    Turn batches of several anchors each where every anchor is represented by
    absolute (x1, y1, x2, y2) values, into batches of the same anchors where
     eachanchor is represented by absolute (x, y, w, h) values - x and y
    represent respectively the x and y coordinates of the top-left corner, w
    and y represent respectively the width and height of sides, x1 and y1
    represent respectively the x and y coordinates of the top-left corner, x2
    and y2 represent respectively the x and y coordinates of the bottom-right
    corner.
    ---
        Input Shape:
            - (
                VARIABLE_N_SAMPLES,
                VARIABLE_N_BOUNDING_BOXES,
                4
            )
    ---
        Output Shape:
            - (
                VARIABLE_N_SAMPLES,
                VARIABLE_N_BOUNDING_BOXES,
                4
            )
    """
    batched_anchors_absolute_x = batched_anchors_absolute_x1_y1_x2_y2[..., 0]
    # shape → (samples, boxes)

    batched_anchors_absolute_y = batched_anchors_absolute_x1_y1_x2_y2[..., 1]
    # shape → (samples, boxes)

    batched_anchors_absolute_w = subtract(
        x=batched_anchors_absolute_x1_y1_x2_y2[..., 2],
        y=batched_anchors_absolute_x1_y1_x2_y2[..., 0]
    )  # shape → (samples, boxes)

    batched_anchors_absolute_h = subtract(
        x=batched_anchors_absolute_x1_y1_x2_y2[..., 3],
        y=batched_anchors_absolute_x1_y1_x2_y2[..., 1]
    )  # shape → (samples, boxes)

    return stack(
        values=(
            batched_anchors_absolute_x,
            batched_anchors_absolute_y,
            batched_anchors_absolute_w,
            batched_anchors_absolute_h
        ),
        axis=-1
    )  # shape → (samples, boxes, 4)


def convert_bounding_boxes_to_final_format(
        batched_bounding_boxes: Tensor,
        batched_n_valid_bounding_boxes: Tensor,
        predicting_online: bool = True
) -> Union[str, List[str]]:
    """
    TODO
     - eventually discretizing all absolute coordinates' values to
    respect the physical constrant of representing image pixels
    ---
        Input Shapes:
            - (
                VARIABLE_N_SAMPLES,
                VARIABLE_N_BOUNDING_BOXES,
                N_OUTPUTS_PER_ANCHOR
            )
            - (
                VARIABLE_N_SAMPLES,
            )
    """
    # if the batched inputs represent a single sample:
    if predicting_online:
        # NOTE: this also automatically asserts that the mini-batch contains
        # only a single sample:
        n_valid_image_bounding_boxes = int(batched_n_valid_bounding_boxes)

        if n_valid_image_bounding_boxes == 0:
            return ''

        image_bounding_boxes = squeeze(
            input=batched_bounding_boxes,
            axis=0
        ).numpy().tolist()[:n_valid_image_bounding_boxes]

        # --------------------------------------------------------------------
        # cast(
        #     # NOTE: rounding is carried out before discretizing so as to
        #     # avoid any truncation due to casting
        #     x=tf_round(
        #         x=...
        #     ),  # shape → (samples, boxes, 4)
        #     dtype=DATA_TYPE_FOR_INPUTS
        # )  # shape → (samples, boxes, 4)
        # --------------------------------------------------------------------

        converted_bounding_boxes = ''
        for index, bounding_box_attributes in enumerate(
                image_bounding_boxes
        ):
            if index != 0:
                converted_bounding_boxes += ' '
            converted_bounding_boxes += (
                '{confidence} {x} {y} {width} {height}'.format(
                    confidence=bounding_box_attributes[0],
                    x=round(bounding_box_attributes[1]),
                    y=round(bounding_box_attributes[2]),
                    width=round(bounding_box_attributes[3]),
                    height=round(bounding_box_attributes[4])
                )
            )

        return converted_bounding_boxes

    else:
            # for image_bounding_boxes, n_valid_image_bounding_boxes in \
            #     batched_bounding_boxes.shape[0]:
        raise NotImplementedError


def get_bounding_boxes_from_model_outputs(
        model_outputs: Tensor,
        from_labels: bool = False
) -> Tuple[Tensor, Tensor]:
    """
    Post-process model outputs by applying format conversion, un-normalization
    and reconstruction, and also non-maximum suppression in case the inputs do
    not intended as labels but as predictions, to turn batched model outputs
    into batches of bounding boxes expressed as (score, x, y, w, h), where x,
    y, w, and h respectively represent the top-lect corner absolute x and y
    coordinates and the absolute width and height, all in pixels - thus as
    (positive or null) integers.

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
    ---
        Output Shapes:
            - (
                VARIABLE_N_SAMPLES,
                VARIABLE_N_BOUNDING_BOXES,
                N_OUTPUTS_PER_ANCHOR
            )
            - (
                VARIABLE_N_SAMPLES,
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
            multiples=(1, 1, 1, N_ANCHORS_PER_CELL, 1)
        ),  # shape → (samples, rows, columns, anchors_per_cell, 2)
        shape=(n_mini_batch_samples, -1, 2)
    )  # shape → (samples, anchors_per_image, 2)

    # applying non-maximum suppression to generate robust bounding box
    # candidates with respective reliability scores when the model outputs
    # are intended as predictions - non-maximum suppression is not relevant
    # when the model outputs are intended as labels as they are already
    # discretized:

    # adding a dummy class dimension for the later Tensorflow's function
    # application - NOTE: a single class in considered in the task of
    # interest:
    anchors_outputs = expand_dims(input=anchors_outputs, axis=2)
    # shape → (samples, anchors_per_image, 1, attributes)

    anchors_scores = anchors_outputs[..., 0]
    # shape → (samples, anchors_per_image, 1)

    anchors_relative_x_y_w_h = anchors_outputs[..., 1:]
    # shape → (samples, anchors_per_image, 1, 4)

    anchors_absolute_x_y_w_h = batched_anchors_rel_to_real_abs_x_y_w_h(
        batched_anchors_relative_x_y_w_h=anchors_relative_x_y_w_h,
        batched_anchors_corners_absolute_x_y=anchors_corners_absolute_x_y
    )  # shape → (samples, anchors_per_image, 1, 4)

    anchors_absolute_x1_y1_x2_y2 = batched_anchors_x_y_w_h_to_x1_y1_x2_y2(
        batched_anchors_absolute_x_y_w_h=anchors_absolute_x_y_w_h
    )  # shape → (samples, anchors_per_image, 1, 4)

    (
        boxes_absolute_x1_y1_x2_y2,  # shape → (samples, boxes, 4)
        boxes_scores,  # shape → (samples, boxes)
        _,  # classes of boxes for each sample, not relevant here
        n_valid_bounding_boxes  # shape → (samples,)
    ) = combined_non_max_suppression(
        boxes=anchors_absolute_x1_y1_x2_y2,
        scores=anchors_scores,
        # NOTE: a single class in considered in the task of interest:
        max_output_size_per_class=MAXIMUM_N_BOUNDING_BOXES_AFTER_NMS,
        max_total_size=MAXIMUM_N_BOUNDING_BOXES_AFTER_NMS,
        iou_threshold=(
            IOU_THRESHOLD_FOR_NON_MAXIMUM_SUPPRESSION if not from_labels
            else 0
        ),
        score_threshold=(
            SCORE_THRESHOLD_FOR_NON_MAXIMUM_SUPPRESSION if not from_labels
            else (1 - 1e-6)
        ),
        pad_per_class=False,
        clip_boxes=False
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
    )  # shape → (samples, boxes, attributes)

    return (
        bounding_boxes_scores_plus_absolute_x_y_w_h,
        # shape → (samples, boxes, attributes)
        n_valid_bounding_boxes
        # shape → (samples,)
    )


if __name__ == '__main__':
    (
        training_samples_and_labels, validation_samples_and_labels
    ) = split_dataset_into_batched_training_and_validation_sets(
        training_plus_validation_set=dataset_of_samples_and_model_outputs(
            shuffle=False
        )
    )

    model = YOLOv3Variant()

    for samples_and_labels in training_samples_and_labels:
        print('\n' + '-'*90)

        (
            expected_bounding_boxes,
            n_valid_expected_bounding_boxes
        ) = get_bounding_boxes_from_model_outputs(
            model_outputs=samples_and_labels[1],
            from_labels=True
        )
        print(
            expected_bounding_boxes.shape,
            '-',
            n_valid_expected_bounding_boxes.shape
        )

        predictions = model(samples_and_labels[0])

        (
            inferred_bounding_boxes,
            n_valid_inferred_bounding_boxes
        ) = get_bounding_boxes_from_model_outputs(
            model_outputs=predictions,
            from_labels=False
        )
        print(
            inferred_bounding_boxes.shape,
            '-',
            n_valid_inferred_bounding_boxes.shape
        )

        break

    print('\n' + '_'*120)

    training_samples_and_labels = (
        training_samples_and_labels.unbatch().batch(1)
    )
    for samples_and_labels in training_samples_and_labels:
        print('\n' + '-'*90)

        (
            expected_bounding_boxes,
            n_valid_expected_bounding_boxes
        ) = get_bounding_boxes_from_model_outputs(
            model_outputs=samples_and_labels[1],
            from_labels=True
        )
        print(
            expected_bounding_boxes.shape,
            '-',
            n_valid_expected_bounding_boxes.shape
        )

        submissions = convert_bounding_boxes_to_final_format(
            batched_bounding_boxes=expected_bounding_boxes,
            batched_n_valid_bounding_boxes=n_valid_expected_bounding_boxes
        )
        print(submissions)

        # predictions = model(samples_and_labels[0])

        # (
        #     inferred_bounding_boxes,
        #     n_valid_inferred_bounding_boxes
        # ) = get_bounding_boxes_from_model_outputs(
        #     model_outputs=predictions,
        #     from_labels=False
        # )
        # print(
        #     inferred_bounding_boxes.shape,
        #     '-',
        #     n_valid_inferred_bounding_boxes.shape
        # )

        # submissions = convert_bounding_boxes_to_final_format(
        #     batched_bounding_boxes=inferred_bounding_boxes,
        #     batched_n_valid_bounding_boxes=n_valid_inferred_bounding_boxes
        # )
        # print(submissions)

        # break
