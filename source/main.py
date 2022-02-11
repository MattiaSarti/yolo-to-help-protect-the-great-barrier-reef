"""
Execution of the proposed competition solution.
"""


from random import seed as random_seed
from time import time

from numpy.random import seed as numpy_seed
# pylint: disable=import-error,no-name-in-module
from tensorflow import convert_to_tensor, expand_dims
from tensorflow.keras import Model
from tensorflow.random import set_seed
# pylint: enable=import-error,no-name-in-module

# only when running everything in a unified notebook on Kaggle's servers:
if __name__ != 'main_by_mattia':
    from common_constants import DATA_TYPE_FOR_INPUTS
    from inference import (
        convert_batched_bounding_boxes_to_final_format,
        get_bounding_boxes_from_model_outputs
    )
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


def infer_on_test_set_and_submit(trained_model_instance: Model) -> None:
    """
    Predict bounding boxes on all test set images, while submitting
    predictions, in an online fashione: one sample at a time.
    NOTE: the logic is the same as specified in the Kaggle competition's
    rules.
    NOTE: the 'pixel_array's served by the competition API iterator are Numpy
    arrays with shape (720, 1280, 3), thus a single sample at a time is
    served, actually having to predict online.
    """
    import greatbarrierreef  # noqa: E501 pylint: disable=import-outside-toplevel,import-error

    # initialize the environment:
    env = greatbarrierreef.make_env()

    # an iterator which loops over the test set and sample submission:
    iter_test = env.iter_test()

    for (pixel_array, sample_prediction_df) in iter_test:
        sample_prediction_df['annotations'] = (  # make your predictions here
            convert_batched_bounding_boxes_to_final_format(
                *(
                    get_bounding_boxes_from_model_outputs(
                        model_outputs=trained_model_instance(
                            expand_dims(
                                input=convert_to_tensor(
                                    value=pixel_array,
                                    dtype=DATA_TYPE_FOR_INPUTS
                                ),
                                axis=0
                            )
                        ),
                        from_labels=False
                    )
                ),
                predicting_online=True,
                as_strings=True
            )
        )
        env.predict(sample_prediction_df)   # register your predictions


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

    model = create_model()

    train_and_validate_model(
        model_instance=model,
        training_set=training_samples_and_labels,
        validation_set=validation_samples_and_labels
    )

    infer_on_test_set_and_submit(trained_model_instance=model)


def time_it(description: str):
    """
    Monitor the execution time (rounded to minutes) of the wrapped function,
    printing the result together with the input description.
    """
    def time_it_with_hardcoded_description(wrapped_function):
        """
        Monitor the execution time (rounded to minutes) of the wrapped
        function, printing the result together with the passed, hardcoded
        description.
        """
        def wrapper(*args, **kwargs):
            """
            Wrap the function to be executed, here hardcoded, with the timing
            and printing operations required.
            """
            tic = time()

            result = wrapped_function(*args, **kwargs)

            toc = time()
            print(
                '\n\t' + description +
                f' <- {round((toc - tic) / 60)} minutes\n'
            )

            return result

        return wrapper

    return time_it_with_hardcoded_description


# adding decorators to time the following functions:
(
    split_dataset_into_batched_training_and_validation_sets
) = time_it(description="DATASETS INITIALIZED")(
    split_dataset_into_batched_training_and_validation_sets
)
create_model = time_it(description="MODEL INITIALIZED")(
    lambda: YOLOv3Variant()
)
train_and_validate_model = time_it(description="MODEL TRAINED AND VALIDATED")(
    train_and_validate_model
)
infer_on_test_set_and_submit = time_it(description="PREDICTIONS MADE")(
    infer_on_test_set_and_submit
)


# only when running everything in a unified notebook on Kaggle's servers:
if __name__ == 'main_by_mattia':
    main()
