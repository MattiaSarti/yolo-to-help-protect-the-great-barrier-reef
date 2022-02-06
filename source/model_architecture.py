"""
Model architecture definition.
"""


# pylint: disable=import-error
from tensorflow import Tensor
from tensorflow.keras import Input, Model, Sequential
from tensorflow.keras.activations import sigmoid
from tensorflow.keras.layers import (
    BatchNormalization,
    Convolution2D,
    LeakyReLU,
    MaxPooling2D,
    Reshape
)
from tensorflow.keras.layers.experimental.preprocessing import (
    RandomFlip,
    Rescaling
)
# pylint: enable=import-error

if __name__ != 'main_by_mattia':
    from common_constants import (
        DOWNSAMPLING_STEPS,
        IMAGE_N_CHANNELS,
        IMAGE_N_COLUMNS,
        IMAGE_N_ROWS,
        N_ANCHORS_PER_CELL,
        N_OUTPUTS_PER_ANCHOR,
        OUTPUT_GRID_N_COLUMNS,
        OUTPUT_GRID_N_ROWS
    )
    from inference import get_bounding_boxes_from_model_outputs


CONVOLUTIONAL_LAYERS_COMMON_KWARGS = {
    'kernel_size': (3, 3),
    'strides': (1, 1),
    'padding': 'same',
    'data_format': 'channels_last',
    'dilation_rate': (1, 1),
    'groups': 1,
    'activation': None,
    'use_bias': True
}
FIRST_LAYER_N_CONVOLUTIONAL_FILTERS = 16  # TODO
INPUT_NORMALIZATION_OFFSET = 0.0
INPUT_NORMALIZATION_RESCALING_FACTOR = (1. / 255)
LEAKY_RELU_NEGATIVE_SLOPE = 0.1
N_CONVOLUTIONS_AT_SAME_RESOLUTION = 3
POOLING_LAYERS_COMMON_KWARGS = {
    'pool_size': (2, 2),
    'strides': (2, 2),
    'padding': 'valid',
    'data_format': 'channels_last',
}


class YOLOv3Variant(Model):  # noqa: E501 pylint: disable=abstract-method, too-many-ancestors
    """
    Customized architecture variant of YOLOv3.
    """

    @staticmethod
    def conv_plus_norm_plus_activation(
            n_of_filters: int
    ) -> Sequential:
        """
        Return an instance of an enriched convolutional layer block composed,
        going from inputs to outputs, of:
        - a 2D convolutional layer without any non-linearity;
        - a batch-normalization layer;
        - a leaky rectified linear unit activation function.
        """
        return Sequential(
            [
                Convolution2D(
                    filters=n_of_filters,
                    **CONVOLUTIONAL_LAYERS_COMMON_KWARGS
                ),
                BatchNormalization(),
                LeakyReLU(
                    alpha=LEAKY_RELU_NEGATIVE_SLOPE
                )
            ]
        )

    @staticmethod
    def build_fully_convolutional_yolov3_architecture() -> Model:
        """
        Return an instance of the herein defined YOLOv3 model architecture
        that represents its fully-convolutional part, that is excluding
        bounding boxes' postprocessing (filtering & aggregation).
        """
        inputs = Input(
            shape=(IMAGE_N_ROWS, IMAGE_N_COLUMNS, IMAGE_N_CHANNELS)
        )

        # rescaling the input image to normalize its pixels' intensities:
        outputs = Rescaling(
            scale=INPUT_NORMALIZATION_RESCALING_FACTOR,
            offset=INPUT_NORMALIZATION_OFFSET
        )(inputs)

        # randomly flipping input images horizontally as a form of data
        # augmentation during training:
        outputs = RandomFlip(mode='horizontal', seed=0,)(outputs)
        # NOTE: step carried out here to take advantage of GPU acceleration,
        # unlike as if it were in the training dataset

        current_n_of_filters = FIRST_LAYER_N_CONVOLUTIONAL_FILTERS
        # for each iso-resolution block of convolutional processing ended by a
        # downsampling:
        for _ in range(DOWNSAMPLING_STEPS):
            # for each enriched convolutional layer in the current
            # iso-resolution block:
            for _ in range(N_CONVOLUTIONS_AT_SAME_RESOLUTION):
                outputs = YOLOv3Variant.conv_plus_norm_plus_activation(
                    n_of_filters=current_n_of_filters
                )(outputs)

            # downsampling, ending the iso-resolution block:
            outputs = MaxPooling2D(**POOLING_LAYERS_COMMON_KWARGS)(outputs)

            # updating the number of filters for the next iso-resolution
            # convolutional layers (by doubling them):
            current_n_of_filters *= 2

        # final 1x1 convolutions to predict bounding boxes' attributes from
        # grid anchors' feature maps:
        outputs = Convolution2D(
            filters=(N_ANCHORS_PER_CELL * N_OUTPUTS_PER_ANCHOR),
            **(
                dict(CONVOLUTIONAL_LAYERS_COMMON_KWARGS, kernel_size=(1, 1))
            )
        )(outputs)
        # NOTE: now bounding boxes' attributes respect the order of meaning
        # (object centered probability, x, y, width, height)

        # asserting the correctness of the current outputs' shape:
        assert (
            outputs.shape[1:] == (
                OUTPUT_GRID_N_ROWS,
                OUTPUT_GRID_N_COLUMNS,
                N_ANCHORS_PER_CELL * N_OUTPUTS_PER_ANCHOR
            )
        ), "Unmatched expectations between outputs and labels shape."

        # reshaping the last output dimension to split anchors and their
        # features along two separate dimensions:
        outputs = Reshape(
            target_shape=(
                OUTPUT_GRID_N_ROWS,
                OUTPUT_GRID_N_COLUMNS,
                N_ANCHORS_PER_CELL,
                N_OUTPUTS_PER_ANCHOR
            )
        )(outputs)

        # applying an element-wise sigmoidal activation function as all 5
        # bounding boxes' output attributes must belong to [0;1] range,
        # since they are either probabilities of a single class (the first
        # attribute) or relative coordinates (the second and third one) or
        # relative sizes (the fourth and fifth one):
        outputs = sigmoid(outputs)
        # NOTE: these sigmoidal computations are carried out here instead of
        # with the loss computation (and during inference) since computing
        # them together with the loss functions's operations would not allow
        # to achieve better gradients during training, since the objectness
        # score needs to undergo the sigmoidal transformation beforehand and
        # the other attributes of the anchors do not udnergo transformations
        # as BCE, that can be fused together with softmax improving gradients'
        # flow, but they all undergo MSE instead, since they represent
        # coordinates and not likelihoods/probabilities

        return Model(
            inputs=inputs,
            outputs=outputs
        )

    def __init__(self) -> None:
        super(YOLOv3Variant, self).__init__()
        self.yolov3_fcn = self.build_fully_convolutional_yolov3_architecture()

    def call(self, inputs: Tensor, training: bool = False) -> Tensor:  # noqa: E501 pylint: disable=arguments-differ
        """
        Forward propagation definition.
        """
        # passing the inputs through the fully-convolutional network:
        fcn_outputs = self.yolov3_fcn(
            inputs=inputs,
            training=training
        )

        # at inference time:
        if not training:
            # post-processing the bounding boxes outputs to return only the
            # final, filtered and aggregated ones:
            get_bounding_boxes_from_model_outputs(
                model_outputs=fcn_outputs,
                from_labels=False
            )

        # at training time:
        else:
            # no post-processing:
            outputs = fcn_outputs

        return outputs


if __name__ == '__main__':
    model = YOLOv3Variant()

    model.yolov3_fcn.summary()
    # TODO: model.plot_model(...)
