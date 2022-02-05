"""
Definitions of the employed loss function and metrics.
"""


# pylint: disable=import-error
from tensorflow import Tensor
from tensorflow.math import reduce_sum
# pylint: enable=import-error


def iou_threshold_averaged_f2_score():
    raise NotImplementedError


def yolov3_variant_loss(y_true: Tensor, y_pred: Tensor) -> Tensor:
    """
    Loss function minimized to train the defined YOLOv3 variant.
    """
    return reduce_sum((y_true - y_pred) ** 2)


if __name__ == '__main__':
    raise NotImplementedError
