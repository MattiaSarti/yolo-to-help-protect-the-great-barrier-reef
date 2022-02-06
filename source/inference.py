"""
Utilities for inference time, for converting model outputs to bounding boxes' predictions.
"""


from typing import List, Tuple

# pylint: disable=import-error
from tensorflow import Tensor
# pylint: enable=import-error


IOU_THRESHOLD_FOR_NON_MAXIMUM_SUPPRESSION = 0.5


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
    TODO
    """
    # when the model outputs are intended as labels:
    if from_labels:
        # non-maximum suppression and the IoU threshold are not relevant when
        # the model outputs represent labels as they are already discretized:
        pass

    # when the model outputs are intended as predictions:
    else:
        pass

    # tf.image.generate_bounding_box_proposals
    # tf.image.combined_non_max_suppression
    # tf.image.non_max_suppression
        # tf.image.non_max_suppression_overlaps
        # tf.image.non_max_suppression_padded
        # tf.image.non_max_suppression_with_scores
    IOU_THRESHOLD_FOR_NON_MAXIMUM_SUPPRESSION

    raise NotImplementedError
