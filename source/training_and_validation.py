"""
Execution of the defined model training and validation on the respective
preprocessed dataset splits, optimizing the defined loss and monitoring the
metrics of interest.
"""


# pylint: disable=import-error
from source.model_architecture import YOLOv3Variant
from source.sample_and_label_extraction import dataset_of_samples_and_model_outputs
from tensorflow.data import Dataset
from tensorflow.keras import Model
# pylint: enable=import-error

from loss_and_metrics import *  # TODO
from model_architecture import YOLOv3Variant
from sample_and_label_extraction import dataset_of_samples_and_model_outputs


raise NotImplementedError
def train_model(model_instance: Model, dataset: Dataset) -> str:  # TODO: output dtype
    """
    Compile (in TensorFlow's language acception, i.e. associate optimizer,
    loss function and metrics to the model instance) the input model instance
    first and then train it on the input dataset.
    """
    model_instance.compile()

    training_history = model_instance.fit()

    return training_history


if __name__ == '__main__':
    samples_and_labels = dataset_of_samples_and_model_outputs()
    model = YOLOv3Variant()

    training_history = train_model(
        model_instance=model,
        dataset=samples_and_labels
    )
