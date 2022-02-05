"""
Execution of the defined model training and validation on the respective
preprocessed dataset splits, optimizing the defined loss and monitoring the
metrics of interest.
"""


# pylint: disable=import-error
from tensorflow.data import Dataset
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
# pylint: enable=import-error

from loss_and_metrics import (
    iou_threshold_averaged_f2_score, yolov3_variant_loss
)
from model_architecture import YOLOv3Variant
from samples_and_labels import dataset_of_samples_and_model_outputs


LEARNING_RATE = 1e-3
N_EPOCHS = 10


def train_model(model_instance: Model, dataset: Dataset) -> str:  # TODO: output dtype
    """
    Compile (in TensorFlow's language acception, i.e. associate optimizer,
    loss function and metrics to the model instance) the input model instance
    first and then train it on the input dataset.
    """
    model_instance.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss=yolov3_variant_loss,
        metrics=[]
    )

    training_history = model_instance.fit(
        x=dataset,
        epochs=N_EPOCHS
    )

    return training_history


if __name__ == '__main__':
    samples_and_labels = dataset_of_samples_and_model_outputs()
    model = YOLOv3Variant()

    training_history = train_model(
        model_instance=model,
        dataset=samples_and_labels
    )
