"""
Model architecture definition.
"""


# pylint: disable=import-error
from tensorflow import Tensor
from tensorflow.keras import Input, Model
# pylint: enable=import-error

from common_constants import IMAGE_N_ROWS, IMAGE_N_COLUMNS, IMAGE_N_CHANNELS


LEAKY_RELU_NEGATIVE_SLOPE = 0.1


def build_fully_convolutional_yolov3_architecture() -> Model:
    """
    Return an instance of the herein defined YOLOv3 model architecture that
    represents its fully-convolutional part, that is excluding bounding boxes'
    postprocessing (filtering & aggregation).
    """
    inputs = Input(
        shape=(IMAGE_N_ROWS, IMAGE_N_COLUMNS, IMAGE_N_CHANNELS)
    )

    raise NotImplementedError
    # tf.keras.activations.relu(x, alpha=LEAKY_RELU_NEGATIVE_SLOPE)

    # TODO: assert xxx.shape[yyy] == N_ANCHORS

    return Model(
        inputs=inputs,
        outputs=[]
    )


class YOLOv3Variant(Model):  # noqa: E501 pylint: disable=abstract-method, too-many-ancestors
    """
    Customized architecture variant of YOLOv3.
    """

    def __init__(self) -> None:
        super(YOLOv3Variant, self).__init__()

        self.yolov3_fcn = build_fully_convolutional_yolov3_architecture()

    def call(self, inputs: Tensor, training: bool = False) -> Tensor:  # noqa: E501 pylint: disable=arguments-differ
        """
        Forward propagation definition.
        """
        # passing the inputs through the fully-convolutional network:
        fcn_outputs = self.yolov3_fcn(
            inputs=inputs,
            training=training
        )

        # post-processing the bounding boxes outputs to return only the final,
        # filtered and aggregated ones:
        raise NotImplementedError

        return fcn_outputs


if __name__ == '__main__':
    model = YOLOv3Variant()

    model.summary()
    # model.plot_model(...)
