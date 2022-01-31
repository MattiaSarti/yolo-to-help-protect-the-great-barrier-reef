"""
Model architecture definition.
"""


# pylint: disable=import-error
from tensorflow import Tensor
from tensorflow.keras import Input, Model
# pylint: enable=import-error


IMAGE_N_ROWS = 720
IMAGE_N_COLUMNS = 1280
IMAGE_N_CHANNELS = 3

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
    #tf.keras.activations.relu(x, alpha=LEAKY_RELU_NEGATIVE_SLOPE)

    # TODO: assert xxx.shape[yyy] == N_ANCHORS

    return Model(
        inputs=inputs,
        outputs=[]
    )


class YOLOv3Variant(Model):
    """
    Customized architecture variant of YOLOv3.
    """

    def __init__(self) -> None:
        super(YOLOv3Variant, self).__init__()

        self.yolov3_fcn = build_fully_convolutional_yolov3_architecture()

    def call(self, inputs: Tensor, training : bool = False) -> Tensor:
        """
        Forward propagation definition.
        """
        # passing the inputs through the fully-convolutional network:
        x = self.yolov3_fcn(
            inputs=inputs,
            training=training
        )

        # post-processing the bounding boxes outputs to return only the final,
        # filtered and aggregated ones:
        raise NotImplementedError

        return x


if __name__ == '__main__':
    model = YOLOv3Variant()

    model.summary()
    # model.plot_model(...)
