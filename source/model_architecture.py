"""
Model architecture definition.
"""


# pylint: disable=import-error
from tensorflow import Tensor
from tensorflow.keras import Input, Model, Sequential
from tensorflow.keras.layers import (
    BatchNormalization,
    Convolution2D,
    Lambda,
    LeakyReLU,
    MaxPooling2D
)
# pylint: enable=import-error

from common_constants import (
    DOWNSAMPLING_STEPS,
    IMAGE_N_CHANNELS,
    IMAGE_N_COLUMNS,
    IMAGE_N_ROWS,
    N_CONVOLUTIONS_AT_SAME_RESOLUTION,
    N_OUTPUTS_PER_ANCHOR,
    OUTPUT_GRID_CELL_N_ANCHORS,
    OUTPUT_GRID_N_COLUMNS,
    OUTPUT_GRID_N_ROWS
)


CONVOLUTIONAL_LAYERS_COMMON_KWARGS = {
    'kernel_size': (3, 3),
    'strides': (1, 1),
    'padding': 'same',  # TODO
    'data_format': 'channels_last',
    'dilation_rate': (1, 1),
    'groups': 1,
    'activation': None,
    'use_bias': True
}
LEAKY_RELU_NEGATIVE_SLOPE = 0.1
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

        outputs = inputs

        # for each iso-resolution block of convolutional processing ended by a
        # downsampling:
        current_n_of_filters = 32
        for _ in range(DOWNSAMPLING_STEPS):
            current_n_of_filters *= 2
            # for each enriched convolutional layer in the current
            # iso-resolution block:
            for _ in range(N_CONVOLUTIONS_AT_SAME_RESOLUTION):
                outputs = YOLOv3Variant.conv_plus_norm_plus_activation(
                    n_of_filters=current_n_of_filters
                )(outputs)

            # downsampling, ending the iso-resolution block:
            outputs = MaxPooling2D(**POOLING_LAYERS_COMMON_KWARGS)(outputs)

        # final 1x1 convolutions to predict bounding boxes' attributes from
        # grid anchors' feature maps:
        outputs = Convolution2D(
            filters=(N_OUTPUTS_PER_ANCHOR * OUTPUT_GRID_CELL_N_ANCHORS),
            **(
                dict(CONVOLUTIONAL_LAYERS_COMMON_KWARGS, kernel_size=(1, 1))
            )
        )(outputs)

        # TODO: outputs = Lambda()(outputs)

        # TODO: outputs = Lambda()(outputs)

        # asserting the correctness of the outputs' shape:
        assert (
            outputs.shape[1:] == (
                OUTPUT_GRID_N_ROWS,
                OUTPUT_GRID_N_COLUMNS,
                OUTPUT_GRID_CELL_N_ANCHORS,
                N_OUTPUTS_PER_ANCHOR
            )
        ), "Unmatched expectations between outputs and labels shape."

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

        if not training:
            # post-processing the bounding boxes outputs to return only the
            # final, filtered and aggregated ones:
            raise NotImplementedError
            # TODO:
            # tf.image.generate_bounding_box_proposals
            # tf.image.combined_non_max_suppression
            # tf.image.non_max_suppression
                # tf.image.non_max_suppression_overlaps
                # tf.image.non_max_suppression_padded
                # tf.image.non_max_suppression_with_scores

        return fcn_outputs


if __name__ == '__main__':
    model = YOLOv3Variant()

    model.yolov3_fcn.summary()
    # TODO: model.plot_model(...)
