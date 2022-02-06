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

if __name__ != 'main_by_mattia':
    from loss_and_metrics import (
        iou_threshold_averaged_f2_score, yolov3_variant_loss
    )
    from model_architecture import YOLOv3Variant
    from samples_and_labels import (
        dataset_of_samples_and_model_outputs,
        split_dataset_into_batched_training_and_validation_sets
    )


LEARNING_RATE = 1e-3
N_EPOCHS = 10

# NOTE: these are 1-based indexes:
EPOCHS_WHEN_VALIDATION_CARRIED_OUT = [1, 3, 5, 7, 9, N_EPOCHS]


def train_and_validate_model(
        model_instance: Model,
        training_set: Dataset,
        validation_set: Dataset
) -> str:  # TODO: output dtype
    """
    Compile (in TensorFlow's language acception, i.e. associate optimizer,
    loss function and metrics to the model instance) the input model instance
    first and then train and validate it on the respective input datasets.
    """
    model_instance.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss=yolov3_variant_loss,
        metrics=[iou_threshold_averaged_f2_score]
    )

    training_history = model_instance.fit(
        x=training_set,
        epochs=N_EPOCHS,
        validation_data=validation_set,
        validation_freq=EPOCHS_WHEN_VALIDATION_CARRIED_OUT
    )

    return training_history


if __name__ == '__main__':
    (
        training_samples_and_labels, validation_samples_and_labels
    ) = split_dataset_into_batched_training_and_validation_sets(
        training_plus_validation_set=dataset_of_samples_and_model_outputs()
    )
    model = YOLOv3Variant()

    training_history = train_and_validate_model(
        model_instance=model,
        training_set=training_samples_and_labels,
        validation_set=validation_samples_and_labels,
    )
