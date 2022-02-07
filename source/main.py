"""
Execution of the proposed competition solution.
"""


from random import seed as random_seed

from numpy.random import seed as numpy_seed
# pylint: disable=import-error
from tensorflow.random import set_seed
# pylint: enable=import-error

if __name__ != 'main_by_mattia':
    from model_architecture import YOLOv3Variant
    from samples_and_labels import (
        dataset_of_samples_and_model_outputs,
        split_dataset_into_batched_training_and_validation_sets
    )
    from training_and_validation import train_and_validate_model


def fix_seeds_for_reproducible_results() -> None:
    """
    Make the subsequent instructions produce purely deterministic outputs by
    fixing all the relevant seeds.
    """
    random_seed(a=0)
    _ = numpy_seed(seed=0)
    set_seed(seed=0)


def main() -> None:
    """
    Execute the proposed competition solution.
    """
    fix_seeds_for_reproducible_results()

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


if __name__ == 'main_by_mattia':
    main()
