"""
Model architecture definition.
"""


# pylint: disable=import-error
from tensorflow.keras import Input, Model
# pylint: enable=import-error


IMAGE_N_ROWS = 720
IMAGE_N_COLUMNS = 1280
IMAGE_N_CHANNELS = 3


def build_model_architecture() -> Model:
    """
    Return an instance of the herein defined model architecture.
    """
    inputs = Input(
        shape=(IMAGE_N_ROWS, IMAGE_N_COLUMNS, IMAGE_N_CHANNELS)
    )

    # ------------------------------------------------------------------------
    # ALTERNATIVE SOLUTION IF MORE FLEXIBILITY IS REQUIRED:
    #
    # class MyModel(tf.keras.Model):
    #     def __init__(self):
    #         super().__init__()
    #         self.dense1 = tf.keras.layers.Dense(4, activation=tf.nn.relu)
    #         self.dense2 = tf.keras.layers.Dense(5, activation=tf.nn.softmax)
    #         self.dropout = tf.keras.layers.Dropout(0.5)
    #     def call(self, inputs, training=False):
    #         x = self.dense1(inputs)
    #         if training:
    #             x = self.dropout(x, training=training)
    #         return self.dense2(x)
    #     model = MyModel()
    # ------------------------------------------------------------------------

    return Model(
        inputs=inputs,
        outputs=[]
    )
