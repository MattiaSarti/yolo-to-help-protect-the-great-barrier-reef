"""
Execution of the defined model training and validation on the respective
preprocessed dataset splits, optimizing the defined loss and monitoring the
metrics of interest.
"""


from os import getcwd, pardir
from os.path import join as path_join
from typing import List

from matplotlib.pyplot import (
    close,
    figure,
    legend,
    pause,
    plot,
    savefig,
    show,
    xlabel,
    ylabel
)
# pylint: disable=import-error,no-name-in-module
from tensorflow.data import Dataset
from tensorflow.keras import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
# pylint: enable=import-error,no-name-in-module

# only when running everything in a unified notebook on Kaggle's servers:
if __name__ != 'main_by_mattia':
    from loss_and_metrics import (
        iou_threshold_averaged_f2_score,
        yolov3_variant_loss
    )
    from model_architecture import build_yolov3_variant_architecture
    from samples_and_labels import (
        dataset_of_samples_and_model_outputs,
        split_dataset_into_batched_training_and_validation_sets
    )


LEARNING_RATE = 1e-3
N_EPOCHS = 10

# NOTE: these are 1-based indexes:
EPOCHS_WHEN_VALIDATION_CARRIED_OUT = [
    # 1,
    round(N_EPOCHS / 2),
    # (N_EPOCHS - 1),
    N_EPOCHS
]

# only when running everything in a unified notebook on Kaggle's servers:
if __name__ != 'main_by_mattia':
    MODEL_PATH = path_join(
        getcwd(),
        'models',
        'model.h5'
    )
    TRAINING_AND_VALIDATION_STATISTICS_DIR = path_join(
        getcwd(),
        pardir,
        'docs',
        'pictures'
    )
else:
    MODEL_PATH = path_join(getcwd(), 'model.h5')
    TRAINING_AND_VALIDATION_STATISTICS_DIR = getcwd()


def plot_and_save_training_and_validation_statistics(
        training_epoch_numbers: List[int],
        training_loss_values: List[float],
        validation_epoch_numbers: List[int],
        validation_loss_values: List[float],
        validation_metric_values: List[float],
) -> None:
    """
    Plot and save the training and validation loss trends and validation
    metric trend with epochs.
    """
    figure()

    plot(
        training_epoch_numbers,
        training_loss_values,
        label='training'
    )
    plot(
        validation_epoch_numbers,
        validation_loss_values,
        'ro',
        label='validation'
    )

    xlabel(xlabel="Epoch Number")
    ylabel(ylabel="Loss")
    legend()

    savefig(
        fname=path_join(
            TRAINING_AND_VALIDATION_STATISTICS_DIR,
            'Training and Validation Loss Trends.png'
        ),
        bbox_inches='tight'
    )

    show(block=False)
    pause(interval=5)

    close()

    figure()

    plot(validation_epoch_numbers, validation_metric_values, 'ro')

    xlabel(xlabel="Epoch Number")
    ylabel(ylabel="Metric")

    savefig(
        fname=path_join(
            TRAINING_AND_VALIDATION_STATISTICS_DIR,
            'Validation Metric Trend.png'
        ),
        bbox_inches='tight'
    )

    show(block=False)
    pause(interval=5)

    close()


def train_and_validate_and_save_model(
        model_instance: Model,
        training_set: Dataset,
        validation_set: Dataset
) -> None:
    """
    Compile (in TensorFlow's language acception, i.e. associate optimizer,
    loss function and metrics to) the input model instance and alternatively
    training and validating it on the respective input datasets, eventually
    plotting and saving training and validation statistics, and saving the
    trained model itself, to the file system.
    """
    # the same optimizer is references throughout all the training procedure
    # so as not to lose its internal states/weights, since it's a stateful
    # optimizer whose parameters are updated during training - as well as the
    # model ones:
    optimizer = Adam(learning_rate=LEARNING_RATE)

    # initializing the training and validation statistics:
    epoch_numbers = []
    training_loss_trend = []
    validation_loss_trend = []
    validation_metric_trend = []

    # for each epoch:
    for epoch_number in range(1, (N_EPOCHS + 1)):
        epoch_numbers.append(epoch_number)

        # training:

        # re-compiling the model to avoid the eager metric computation:
        model_instance.compile(
            optimizer=optimizer,
            loss=yolov3_variant_loss,
            # NOTE: the defined metric cannot be run when not in eager mode,
            # so it is not evaluated while training:
            metrics=[]
        )
        # training the model (on the training set):
        trainin_history = model_instance.fit(
            x=training_set,
            epochs=1,
        )
        # saving the current epoch's training loss value:
        training_loss_trend.append(trainin_history.history['loss'][0])

        if epoch_number in EPOCHS_WHEN_VALIDATION_CARRIED_OUT:
            # validation:

            # re-compiling the model to allow for the eager metric
            # computation:
            model_instance.compile(
                optimizer=optimizer,
                loss=yolov3_variant_loss,
                metrics=[iou_threshold_averaged_f2_score],
                # NOTE: the defined metric can only be run in eager mode:
                run_eagerly=True
            )
            # validating the model (on the validation set):
            loss_and_metric = model_instance.evaluate(
                x=validation_set
            )
            # saving the current epoch's validation loss and metric values:
            validation_loss_trend.append(loss_and_metric[0])
            validation_metric_trend.append(loss_and_metric[1])

    # saving the trained model:
    model_instance.save(
        filepath=MODEL_PATH,
        save_format='h5',  # NOTE: necessary for data augmentation layers
        overwrite=False
    )

    # plotting and saving the training and validation statistics:
    plot_and_save_training_and_validation_statistics(
        training_epoch_numbers=epoch_numbers,
        training_loss_values=training_loss_trend,
        validation_epoch_numbers=EPOCHS_WHEN_VALIDATION_CARRIED_OUT,
        validation_loss_values=validation_loss_trend,
        validation_metric_values=validation_metric_trend,
    )


if __name__ == '__main__':
    (
        training_samples_and_labels, validation_samples_and_labels
    ) = split_dataset_into_batched_training_and_validation_sets(
        training_plus_validation_set=dataset_of_samples_and_model_outputs()
    )

    model = build_yolov3_variant_architecture()

    train_and_validate_and_save_model(
        model_instance=model,
        training_set=training_samples_and_labels.take(4),
        validation_set=validation_samples_and_labels.take(3)
    )

    del model
    trained_model = load_model(
        filepath=MODEL_PATH,
        compile=False
    )
