#pylint: disable=too-many-lines
"""
Sample and label extraction from the raw dataset files, inspection and
preprocessing for feeding the model.
"""


from csv import reader as csv_reader
from itertools import combinations
from json import loads as json_loads
from math import sqrt
from os import getcwd, pardir
from os.path import join as path_join
from random import shuffle
from typing import Dict, List, Tuple

from matplotlib.patches import Rectangle
from matplotlib.pyplot import (
    clf as plt_clf,
    close as plt_close,
    figure as plt_figure,
    hist as plt_hist,
    get_current_fig_manager,
    pause as plt_pause,
    savefig as plt_savefig,
    show as plt_show,
    subplots,
    title as plt_title,
    xticks as plt_xticks
)
from numpy import argmin, sum as np_sum, unravel_index, zeros
# pylint: disable=import-error,no-name-in-module
from tensorflow import convert_to_tensor, py_function, Tensor
from tensorflow.data import AUTOTUNE, Dataset
from tensorflow.io import decode_jpeg, read_file
# pylint: enable=import-error,no-name-in-module

# only when running everything in a unified notebook on Kaggle's servers:
if __name__ != 'main_by_mattia':
    from common_constants import (
        ANCHORS_WIDTH_VS_HEIGHT_WEIGHTS,
        DATA_TYPE_FOR_INPUTS,
        DATA_TYPE_FOR_OUTPUTS,
        IMAGE_N_COLUMNS,
        IMAGE_N_ROWS,
        N_ANCHORS_PER_CELL,
        N_OUTPUTS_PER_ANCHOR,
        OUTPUT_GRID_CELL_CENTERS_XY_COORDS,
        OUTPUT_GRID_CELL_CORNERS_XY_COORDS,
        OUTPUT_GRID_CELL_N_COLUMNS,
        OUTPUT_GRID_CELL_N_ROWS,
        OUTPUT_GRID_N_COLUMNS,
        OUTPUT_GRID_N_ROWS
    )


MINI_BATCH_SIZE = 4  # TODO
VALIDATION_SET_PORTION_OF_DATA = 0.1  # 0.3

# only when running everything in a unified notebook on Kaggle's servers:
if __name__ != 'main_by_mattia':
    DATASET_DIR = path_join(
        getcwd(),
        pardir,
        'tensorflow-great-barrier-reef'
    )
else:
    DATASET_DIR = path_join(
        getcwd(),
        pardir,
        'input',
        'tensorflow-great-barrier-reef'
    )
CACHE_DIR = path_join(
    getcwd(),
    'cache'
)
CACHE_FILE_PATH_FOR_STATISTICS_SET = path_join(
    CACHE_DIR,
    'statistics.tmp'
)
CACHE_FILE_PATH_FOR_TRAINING_SET = path_join(
    CACHE_DIR,
    'training.tmp'
)
CACHE_FILE_PATH_FOR_VALIDATION_SET = path_join(
    CACHE_DIR,
    'validation.tmp'
)
LABELS_FILE_PATH = path_join(
    DATASET_DIR,
    'train.csv'
)
PICTURES_DIR = path_join(
    getcwd(),
    pardir,
    'docs',
    'pictures'
)

SHOW_BOUNDING_BOXES_STATISTICS = False
SHOW_DATASET_MOVIES = False


def get_cell_containing_bounding_box_center(
        center_absolute_x_and_y_coords: Tuple[float, float]
) -> Tuple[int, int, int, int]:
    """
    Find the output grid cell whose center is closest to the bounding box one
    (the input one), returning the grid cell's row and column indexes and its
    top-left corner x and y coordinates.
    ---
        Output Shape:
            - (4,)
    ---
        Output Meaning:
            - [
                grid cell row index,
                grid cell column index,
                x coordindate of cell top-left corner,
                y coordindate of cell top-left corner
            ]
    """
    (  # pylint: disable=unbalanced-tuple-unpacking
        grid_cell_enclosing_bounding_box_center_row_index,
        grid_cell_enclosing_bounding_box_center_column_index
    ) = unravel_index(
        indices=argmin(
            # NOTE: in case of equivalent minima, the first one is picked grid
            # of squared element-wise center pairs' distances representing the
            # minimized objective to find the closest grid cell center:
            a=np_sum(
                a=(
                    (OUTPUT_GRID_CELL_CENTERS_XY_COORDS -
                     center_absolute_x_and_y_coords) ** 2
                ),
                axis=-1
            )
        ),
        shape=(OUTPUT_GRID_N_ROWS, OUTPUT_GRID_N_COLUMNS),
        order='C'
    )

    return (
        # [grid cell row index, grid cell column index]:
        [
            grid_cell_enclosing_bounding_box_center_row_index,
            grid_cell_enclosing_bounding_box_center_column_index
        ] +
        # [x coordindate of cell corner, y coordindate of cell corner]:
        OUTPUT_GRID_CELL_CORNERS_XY_COORDS[
            grid_cell_enclosing_bounding_box_center_row_index,
            grid_cell_enclosing_bounding_box_center_column_index,
            :
        ].tolist()
    )


def get_index_of_anchor_with_closest_aspect_ratio(
        absolute_width: float,
        absolute_height: float
) -> int:
    """
    Return the index of the anchor whose aspect ratio is close to the
    considered bounding box represented by the input relative width and
    relative height.
    """
    width_weight = absolute_width / (absolute_width + absolute_height)
    height_weight = absolute_height / (absolute_width + absolute_height)

    return (
        ANCHORS_WIDTH_VS_HEIGHT_WEIGHTS.index(
            sorted(
                ANCHORS_WIDTH_VS_HEIGHT_WEIGHTS,
                key=lambda width_vs_height_weights: (
                    abs(width_vs_height_weights[0] - width_weight) +
                    abs(width_vs_height_weights[1] - height_weight)
                ),
                reverse=False
            )[0]
        )
    )


def dataset_of_samples_and_bounding_boxes() -> Dataset:
    """
    Build a TensorFlow dataset that can iterate over all the dataset samples
    and the respective labels containing bounding boxes.
    """
    image_paths_dataset = Dataset.from_tensor_slices(
        tensors=[*IMAGE_PATHS_TO_BOUNDING_BOXES]  # only keys included
    )

    image_paths_dataset = image_paths_dataset.map(
        map_func=lambda image_path: py_function(
            func=load_sample_and_get_bounding_boxes,
            inp=[image_path],
            Tout=(DATA_TYPE_FOR_INPUTS, DATA_TYPE_FOR_OUTPUTS)
        ),
        num_parallel_calls=AUTOTUNE,
        deterministic=True
    )

    if __name__ == 'main_by_mattia':
        # optimizing performances by caching end-results:
        image_paths_dataset = image_paths_dataset.cache(
            filename=CACHE_FILE_PATH_FOR_STATISTICS_SET
        )

    # optimizing performances by pre-fetching final elements:
    image_paths_dataset = image_paths_dataset.prefetch(buffer_size=AUTOTUNE)

    return image_paths_dataset


def dataset_of_samples_and_model_outputs(shuffle: bool = True) -> Dataset:
    """
    Build a TensorFlow dataset that can iterate over all the dataset samples
    and the respective labels containing model outputs, in a shuffled order.
    """
    image_paths = [*IMAGE_PATHS_TO_MODEL_OUTPUTS]  # only keys included

    # shuffling is carried out beforehand so as to prevent TensorFlow's
    # dataset elements from being shuffled entirely in memory - shuffling
    # happens before the TensorFlow's dataset creation itself, this way:
    if shuffle:
        shuffle(image_paths)

    image_paths_dataset = Dataset.from_tensor_slices(
        tensors=image_paths
    )

    # FIXME: OOM

    # # NOTE: shuffling is carried out here to have acceptable performance
    # # with a shuffling buffer size that allows to take the whole set into
    # # memory in case shuffling is desired:
    # if shuffle:
    #     image_paths_dataset = image_paths_dataset.shuffle(
    #         buffer_size=N_TRAINING_PLUS_VALIDATION_SAMPLES,
    #         seed=0,
    #         reshuffle_each_iteration=False  # NOTE: relevant when splitting
    #     )

    # NOTE: further optimizations on this dataset - that is the one employed
    # for training/validation - are carried out later, after
    # training/validation splitting and batching, to optimize performances

    return image_paths_dataset.map(
        map_func=lambda image_path: py_function(
            func=load_sample_and_get_model_outputs,
            inp=[image_path],
            Tout=(DATA_TYPE_FOR_INPUTS, DATA_TYPE_FOR_OUTPUTS)
        ),
        num_parallel_calls=AUTOTUNE,
        deterministic=True
    )


def inspect_bounding_boxes_statistics_on_training_n_validation_set() -> None:  # noqa: E501 pylint: disable=too-many-locals,too-many-branches,too-many-statements
    """
    Inspect and print the following statistics of bounding boxes in the
    training-plus-validation set:
        - total number of bounding boxes
        - total number of images
        - average number of bounding boxes per image
        - minimum number of bounding boxes per image
        - maximum number of bounding boxes per image
        - total number of empty images
        - average bounding box height [pixels]
        - average bounding box width [pixels]
        - average bounding boxes' centers distance [pixels]
        - average bounding boxes' centers x-coord distance [pixels]
        - average bounding boxes' centers y-coord distance [pixels]
        - minimum bounding box height [pixels]
        - minimum bounding box width [pixels]
        - minimum bounding boxes' centers distance [pixels]
        - minimum bounding boxes' centers x-coord distance [pixels]
        - minimum bounding boxes' centers y-coord distance [pixels]
        - maximum bounding box height [pixels]
        - maximum bounding box width [pixels]
        - maximum bounding boxes' centers distance [pixels]
        - maximum bounding boxes' centers x-coord distance [pixels]
        - maximum bounding boxes' centers y-coord distance [pixels]
        - histogram of number of bounding boxes per image
        - histogram of bounding boxes' centers distance [pixels]
        - histogram of bounding boxes' centers x-coord distance [pixels]
        - histogram of bounding boxes' centers y-coord distance [pixels]
    """
    total_n_images = len(IMAGE_PATHS_TO_BOUNDING_BOXES)

    bounding_boxes_centers_distances_for_histogram = []
    bounding_boxes_centers_x_coord_distances_for_histogram = []
    bounding_boxes_centers_y_coord_distances_for_histogram = []
    cumulative_bounding_box_height = 0
    cumulative_bounding_box_width = 0
    cumulative_bounding_boxes_centers_distance = 0
    cumulative_bounding_boxes_centers_x_coord_distance = 0
    cumulative_bounding_boxes_centers_y_coord_distance = 0
    minimum_bounding_box_height = 99999
    minimum_bounding_box_width = 99999
    minimum_bounding_boxes_centers_distance = 99999
    minimum_bounding_boxes_centers_x_coord_distance = 99999
    minimum_bounding_boxes_centers_y_coord_distance = 99999
    minimum_n_bounding_boxes_per_image = 99999
    maximum_bounding_box_height = 0
    maximum_bounding_box_width = 0
    maximum_bounding_boxes_centers_distance = 0
    maximum_bounding_boxes_centers_x_coord_distance = 0
    maximum_bounding_boxes_centers_y_coord_distance = 0
    maximum_n_bounding_boxes_per_image = 0
    n_bounding_boxes_per_image_for_histogram = []
    total_n_bounding_boxes = 0
    total_n_bounding_boxes_center_distances_cumulated = 0
    total_n_empty_images = 0

    for image_bounding_boxes in IMAGE_PATHS_TO_BOUNDING_BOXES.values():
        n_bounding_boxes = len(image_bounding_boxes)
        n_bounding_boxes_per_image_for_histogram.append(
            n_bounding_boxes
        )

        total_n_bounding_boxes += n_bounding_boxes
        if n_bounding_boxes < minimum_n_bounding_boxes_per_image:
            minimum_n_bounding_boxes_per_image = n_bounding_boxes
        if n_bounding_boxes > maximum_n_bounding_boxes_per_image:
            maximum_n_bounding_boxes_per_image = n_bounding_boxes
        if n_bounding_boxes == 0:
            total_n_empty_images += 1

        bounding_boxes_centers_x_and_y_coords = []
        for bounding_box in image_bounding_boxes:
            cumulative_bounding_box_height += bounding_box['height']
            cumulative_bounding_box_width += bounding_box['width']

            bounding_boxes_centers_x_and_y_coords.append(
                {
                    'x': (bounding_box['x'] + bounding_box['width']) / 2,
                    'y': (bounding_box['y'] + bounding_box['height']) / 2
                }
            )

            if bounding_box['height'] < minimum_bounding_box_height:
                minimum_bounding_box_height = bounding_box['height']
            if bounding_box['width'] < minimum_bounding_box_width:
                minimum_bounding_box_width = bounding_box['width']

            if bounding_box['height'] > maximum_bounding_box_height:
                maximum_bounding_box_height = bounding_box['height']
            if bounding_box['width'] > maximum_bounding_box_width:
                maximum_bounding_box_width = bounding_box['width']

        if n_bounding_boxes > 1:
            for centers_coords_pair in combinations(
                    iterable=bounding_boxes_centers_x_and_y_coords,
                    r=2
            ):
                total_n_bounding_boxes_center_distances_cumulated += 1

                x_coord_difference = abs(
                    centers_coords_pair[0]['x'] - centers_coords_pair[1]['x']
                )
                y_coord_difference = abs(
                    centers_coords_pair[0]['y'] - centers_coords_pair[1]['y']
                )
                distance = sqrt(
                    x_coord_difference**2 + y_coord_difference**2
                )

                bounding_boxes_centers_distances_for_histogram.append(
                    distance
                )
                bounding_boxes_centers_x_coord_distances_for_histogram.append(
                    x_coord_difference
                )
                bounding_boxes_centers_y_coord_distances_for_histogram.append(
                    y_coord_difference
                )

                cumulative_bounding_boxes_centers_distance += (
                    distance
                )
                cumulative_bounding_boxes_centers_x_coord_distance += (
                    x_coord_difference
                )
                cumulative_bounding_boxes_centers_y_coord_distance += (
                    y_coord_difference
                )

                if (
                        distance <
                        minimum_bounding_boxes_centers_distance
                ):
                    minimum_bounding_boxes_centers_distance = (
                        distance
                    )
                if (
                        x_coord_difference <
                        minimum_bounding_boxes_centers_x_coord_distance
                ):
                    minimum_bounding_boxes_centers_x_coord_distance = (
                        x_coord_difference
                    )
                if (
                        y_coord_difference <
                        minimum_bounding_boxes_centers_y_coord_distance
                ):
                    minimum_bounding_boxes_centers_y_coord_distance = (
                        y_coord_difference
                    )

                if (
                        distance >
                        maximum_bounding_boxes_centers_distance
                ):
                    maximum_bounding_boxes_centers_distance = (
                        distance
                    )
                if (
                        x_coord_difference >
                        maximum_bounding_boxes_centers_x_coord_distance
                ):
                    maximum_bounding_boxes_centers_x_coord_distance = (
                        x_coord_difference
                    )
                if (
                        y_coord_difference > (
                            maximum_bounding_boxes_centers_y_coord_distance
                        )
                ):
                    maximum_bounding_boxes_centers_y_coord_distance = (
                        y_coord_difference
                    )

    print('- ' * 30)
    print("Bounding Boxes' Statistics:")

    print(
        "\t- total number of bounding boxes:",
        total_n_bounding_boxes
    )
    print(
        "\t- total number of images:",
        total_n_images
    )
    print(
        "\t- average number of bounding boxes per image:",
        round(number=total_n_bounding_boxes/total_n_images, ndigits=2)
    )
    print(
        "\t- minimum number of bounding boxes per image:",
        minimum_n_bounding_boxes_per_image
    )
    print(
        "\t- maximum number of bounding boxes per image:",
        maximum_n_bounding_boxes_per_image
    )
    print(
        "\t- total number of empty images:",
        total_n_empty_images
    )
    print(
        "\t- average bounding box height [pixels]:",
        round(
            number=cumulative_bounding_box_height/total_n_bounding_boxes,
            ndigits=2
        )
    )
    print(
        "\t- average bounding box width [pixels]:",
        round(
            number=cumulative_bounding_box_width/total_n_bounding_boxes,
            ndigits=2
        )
    )
    print(
        "\t- average bounding boxes' centers distance [pixels]:",
        round(
            number=(
                cumulative_bounding_boxes_centers_distance /
                total_n_bounding_boxes_center_distances_cumulated
            ),
            ndigits=2
        )
    )
    print(
        "\t- average bounding boxes' centers x-coord distance [pixels]:",
        round(
            number=(
                cumulative_bounding_boxes_centers_x_coord_distance /
                total_n_bounding_boxes_center_distances_cumulated
            ),
            ndigits=2
        )
    )
    print(
        "\t- average bounding boxes' centers y-coord distance [pixels]:",
        round(
            number=(
                cumulative_bounding_boxes_centers_y_coord_distance /
                total_n_bounding_boxes_center_distances_cumulated
            ),
            ndigits=2
        )
    )
    print(
        "\t- minimum bounding box height [pixels]:",
        minimum_bounding_box_height
    )
    print(
        "\t- minimum bounding box width [pixels]:",
        minimum_bounding_box_width
    )
    print(
        "\t- minimum bounding boxes' centers distance [pixels]:",
        round(
            number=minimum_bounding_boxes_centers_distance,
            ndigits=2
        )
    )
    print(
        "\t- minimum bounding boxes' centers x-coord distance [pixels]:",
        minimum_bounding_boxes_centers_x_coord_distance
    )
    print(
        "\t- minimum bounding boxes' centers y-coord distance [pixels]:",
        minimum_bounding_boxes_centers_y_coord_distance
    )
    print(
        "\t- maximum bounding box height [pixels]:",
        maximum_bounding_box_height
    )
    print(
        "\t- maximum bounding box width [pixels]:",
        maximum_bounding_box_width
    )
    print(
        "\t- maximum bounding boxes' centers distance [pixels]:",
        round(
            number=maximum_bounding_boxes_centers_distance,
            ndigits=2
        )
    )
    print(
        "\t- maximum bounding boxes' centers x-coord distance [pixels]:",
        maximum_bounding_boxes_centers_x_coord_distance
    )
    print(
        "\t- maximum bounding boxes' centers y-coord distance [pixels]:",
        maximum_bounding_boxes_centers_y_coord_distance
    )
    print(
        "\t- histogram of number of bounding boxes per image: see plot"
    )
    print(
        "\t- histogram of bounding boxes' centers distance [pixels]: " +
        "see plot"
    )
    print(
        "\t- histogram of bounding boxes' centers x-coord distance " +
        "[pixels]: see plot"
    )
    print(
        "\t- histogram of bounding boxes' centers y-coord distance " +
        "[pixels]: see plot"
    )

    plt_figure()

    what_it_represent = "Histogram of Number of Bounding Boxes per Image"
    plt_hist(
        x=n_bounding_boxes_per_image_for_histogram,
        bins=maximum_n_bounding_boxes_per_image,
        align='left',
        color='skyblue',
        rwidth=0.8
    )
    plt_title(label=what_it_represent)
    plt_xticks(
        ticks=list(range(maximum_n_bounding_boxes_per_image))
    )
    plt_savefig(
        fname=path_join(
            PICTURES_DIR,
            what_it_represent + '.png'
        ),
        bbox_inches='tight'
    )
    plt_show(block=False)
    plt_pause(interval=1)
    plt_clf()

    what_it_represent = (
        "Histogram of Bounding Boxes' Centers Distance [pixels]"
    )
    plt_hist(
        x=bounding_boxes_centers_distances_for_histogram,
        bins=list(range(int(sqrt(IMAGE_N_COLUMNS**2 + IMAGE_N_ROWS**2)))),
        align='left',
        color='chartreuse',
        rwidth=0.8
    )
    plt_title(label=what_it_represent)
    plt_xticks(
        ticks=list(
            range(0, int(sqrt(IMAGE_N_COLUMNS**2 + IMAGE_N_ROWS**2)), 20)
        ),
        fontsize=6,
        rotation=90
    )
    figure_manager = get_current_fig_manager()
    figure_manager.resize(*figure_manager.window.maxsize())
    plt_savefig(
        fname=path_join(
            PICTURES_DIR,
            what_it_represent + '.png'
        ),
        bbox_inches='tight'
    )
    plt_show(block=False)
    plt_pause(interval=1)
    plt_clf()

    what_it_represent = (
        "Histogram of Bounding Boxes' Centers X-Coordinate Distance [pixels]"
    )
    plt_hist(
        x=bounding_boxes_centers_x_coord_distances_for_histogram,
        bins=list(range(IMAGE_N_COLUMNS)),
        align='left',
        color='mediumslateblue',
        rwidth=0.8
    )
    plt_title(label=what_it_represent)
    plt_xticks(
        ticks=list(range(0, IMAGE_N_COLUMNS, 20)),
        fontsize=6,
        rotation=90
    )
    plt_savefig(
        fname=path_join(
            PICTURES_DIR,
            what_it_represent + '.png'
        ),
        bbox_inches='tight'
    )
    plt_show(block=False)
    plt_pause(interval=1)
    plt_clf()

    what_it_represent = (
        "Histogram of Bounding Boxes' Centers Y-Coordinate Distance [pixels]"
    )
    plt_hist(
        x=bounding_boxes_centers_y_coord_distances_for_histogram,
        bins=list(range(IMAGE_N_ROWS)),
        align='left',
        color='violet',
        rwidth=0.8
    )
    plt_title(label=what_it_represent)
    plt_xticks(
        ticks=list(range(0, IMAGE_N_ROWS, 20)),
        fontsize=6,
        rotation=90
    )
    plt_savefig(
        fname=path_join(
            PICTURES_DIR,
            what_it_represent + '.png'
        ),
        bbox_inches='tight'
    )
    plt_show(block=False)
    plt_pause(interval=1)
    plt_clf()

    plt_close()

    print('- ' * 30)


def label_line_to_image_path_2_bounding_boxes_and_2_model_output(
        csv_label_line_segments: List[str]
) -> Tuple[
        Dict[str, List[Dict[str, int]]],
        Dict[str, List[List[Tuple[int, int, int, int]]]]
]:
    """
    Turn any line of the CSV labels file from the original format
    'video_id,sequence,video_frame,sequence_frame,image_id,annotations' into
    two dictionariies: the former with the respective image file path as key
    and the respective bounding boxes as value, the latter with the respective
    image file path as key and the respective model outputs as value.
    """
    image_path = path_join(
        DATASET_DIR,
        'train_images',
        'video_' + csv_label_line_segments[0],
        csv_label_line_segments[2] + '.jpg'
    )
    bounding_boxes = json_loads(
        csv_label_line_segments[5]
        .replace('"', '"""')
        .replace("'", '"')
    )

    return (
        {
            bytes(image_path, 'utf-8'): bounding_boxes
        },
        {
            bytes(image_path, 'utf-8'): turn_bounding_boxes_to_model_outputs(
                raw_bounding_boxes=bounding_boxes
            )
        }
    )


def load_labels_as_paths_to_bounding_boxes_and_model_outputs_dicts() -> Tuple[
        Dict[str, List[Dict[str, int]]],
        Dict[str, List[List[Tuple[int, int, int, int]]]]
]:
    """
    Load the labels' information from the CSV file and return them as a two
    dictionaries, the former associating image file paths to respective
    bounding boxes and the latter associating image file paths to respective
    model outputs.
    """
    image_paths_to_bounding_boxes = {}
    image_paths_to_model_outputs = {}

    with open(LABELS_FILE_PATH, 'r') as labels_file:
        labels_reader = csv_reader(
            labels_file,
            delimiter=',',
            quotechar='"'
        )

        for line_index, line_segments in enumerate(labels_reader):
            if line_index == 0:
                continue

            # turning the label from the raw format into processed
            # dictionaries to retrieve bounding boxes and model outputs of
            # images easily from respective image file paths:
            (
                image_path_to_bounding_boxes,
                image_path_to_model_outputs
            ) = label_line_to_image_path_2_bounding_boxes_and_2_model_output(
                csv_label_line_segments=line_segments
            )
            image_paths_to_bounding_boxes.update(image_path_to_bounding_boxes)
            image_paths_to_model_outputs.update(image_path_to_model_outputs)

    return (image_paths_to_bounding_boxes, image_paths_to_model_outputs)


def load_sample_and_get_bounding_boxes(image_path: Tensor) -> Tuple[
        Tensor, Tensor
]:
    """
    Load the sample and get the label - representing bounding boxes - of the
    image represented by the input path.
    """
    return (
        decode_jpeg(
            contents=read_file(
                filename=image_path
            )
        ),
        convert_to_tensor(
            # bounding boxes as network output values:
            value=[
                [
                    bounding_box_dict['x'],
                    bounding_box_dict['y'],
                    bounding_box_dict['width'],
                    bounding_box_dict['height']
                ] for bounding_box_dict in
                IMAGE_PATHS_TO_BOUNDING_BOXES[image_path.numpy()]
            ],
            dtype=DATA_TYPE_FOR_OUTPUTS
        )
    )


def load_sample_and_get_model_outputs(image_path: Tensor) -> Tuple[
        Tensor, Tensor
]:
    """
    Load the sample and get the label - representing model outputs - of the
    image represented by the input path.
    """
    return (
        decode_jpeg(
            contents=read_file(
                filename=image_path
            )
        ),
        convert_to_tensor(
            # bounding boxes as network output values:
            value=IMAGE_PATHS_TO_MODEL_OUTPUTS[image_path.numpy()],
            dtype=DATA_TYPE_FOR_OUTPUTS
        )
    )


def split_dataset_into_batched_training_and_validation_sets(
        training_plus_validation_set: Dataset
) -> Tuple[Dataset, Dataset]:
    """
    Split the input dataset into a training set and a validation set, both
    already divided into mini-batches.
    """
    n_samples_in_validation_set = int(
        VALIDATION_SET_PORTION_OF_DATA * N_TRAINING_PLUS_VALIDATION_SAMPLES
    )
    n_samples_in_training_set = (
        N_TRAINING_PLUS_VALIDATION_SAMPLES - n_samples_in_validation_set
    )

    # training set:

    training_set = (
        training_plus_validation_set
        # selecting only the training samples and labels:
        .take(count=n_samples_in_training_set)
        # creating mini-batches:
        .batch(
            batch_size=MINI_BATCH_SIZE,
            drop_remainder=False,
            num_parallel_calls=AUTOTUNE,
            deterministic=True
        )
    )

    # only when running everything in a unified notebook on Kaggle's servers:
    if __name__ == 'main_by_mattia':
        # optimizing performances by caching end-results:
        training_set = training_set.cache(
            filename=CACHE_FILE_PATH_FOR_TRAINING_SET
        )

    # optimizing performances by pre-fetching final elements:
    training_set = training_set.prefetch(buffer_size=AUTOTUNE)

    # validation set:

    validation_set = (
        training_plus_validation_set
        # selecting only the validation samples and labels:
        .skip(count=n_samples_in_training_set)
        .take(count=n_samples_in_validation_set)
        # creating mini-batches:
        .batch(
            batch_size=MINI_BATCH_SIZE,
            drop_remainder=False,
            num_parallel_calls=AUTOTUNE,
            deterministic=True
        )
    )

    # only when running everything in a unified notebook on Kaggle's servers:
    if __name__ == 'main_by_mattia':
        # optimizing performances by caching end-results:
        validation_set = validation_set.cache(
            filename=CACHE_FILE_PATH_FOR_TRAINING_SET
        )

    # optimizing performances by pre-fetching final elements:
    validation_set = validation_set.prefetch(buffer_size=AUTOTUNE)

    return (training_set, validation_set)


def show_dataset_as_movie(
        ordered_samples_and_labels: Dataset,
        bounding_boxes_or_model_outputs: str = 'bounding_boxes'
) -> None:
    """
    Show the dataset images frame by frame, reconstructing the video
    sequences, with boundinx boxes contained displayed over the respective
    sample/frame.
    """
    assert (
        bounding_boxes_or_model_outputs in ('bounding_boxes', 'model_outputs')
    ), "Invalid 'bounding_boxes_or_model_outputs' input."

    _, axes = subplots(1, 1)

    # for each sample-label pair, a frame fusing them together is shown:
    for index, sample_and_label in enumerate(ordered_samples_and_labels):
        if index % 1000 == 0:
            print(f"{index} frames shown")

        # clearing axes from the previous frame information:
        axes.clear()

        # showing the image:
        axes.imshow(sample_and_label[0].numpy())

        # showing labels...

        # ... either as bounding boxes:
        if bounding_boxes_or_model_outputs == 'bounding_boxes':
            # for each bounding box:
            for bounding_box in sample_and_label[1].numpy().tolist():
                # drawing the bounding box over the frame image:
                axes.add_patch(
                    p=Rectangle(
                        xy=(bounding_box[0], bounding_box[1]),
                        width=bounding_box[2],
                        height=bounding_box[3],
                        linewidth=2,
                        edgecolor='#00ff00',
                        facecolor='none'
                    )
                )

        # ... or as model output grid cells:
        elif bounding_boxes_or_model_outputs == 'model_outputs':
            # for each model output grid cell whose label contains anchors:
            for cell_row_index in range(OUTPUT_GRID_N_ROWS):
                for cell_column_index in range(OUTPUT_GRID_N_COLUMNS):
                    # filtering out grid cells not containing any anchor:
                    if (
                            sample_and_label[1][
                                cell_row_index,
                                cell_column_index,
                                :,
                                :
                            ].numpy() == zeros(
                                shape=(
                                    N_ANCHORS_PER_CELL,
                                    N_OUTPUTS_PER_ANCHOR
                                )
                            )
                    ).all():
                        continue

                    # highlighting the full cell over the frame image:
                    axes.add_patch(
                        p=Rectangle(
                            xy=(
                                OUTPUT_GRID_CELL_CORNERS_XY_COORDS[
                                    cell_row_index,
                                    cell_column_index
                                ]
                            ),
                            width=OUTPUT_GRID_CELL_N_COLUMNS,
                            height=OUTPUT_GRID_CELL_N_ROWS,
                            linewidth=2,
                            edgecolor='#00ff00',
                            facecolor='none'
                        )
                    )

        else:
            raise Exception("Ill-conceived code.")

        # making the plot go adeah with the next frame after a small pause for
        # better observation:
        plt_show(block=False)
        plt_pause(interval=0.000001)


def turn_bounding_boxes_to_model_outputs(
        raw_bounding_boxes: List[Dict[str, int]]
) -> Dict[str, List[List[Tuple[int, int, int, int]]]]:
    """
    Turn the input, raw list of bounding boxes' position information into the
    equivalent information from the model outputs' perspective, as direct
    supervision labels - for a single image.
    """
    labels = zeros(
        shape=(
            OUTPUT_GRID_N_ROWS,
            OUTPUT_GRID_N_COLUMNS,
            N_ANCHORS_PER_CELL,
            N_OUTPUTS_PER_ANCHOR
        )
    )

    # for each bounding box in the image:
    for bounding_box in raw_bounding_boxes:
        # computing the absolute x and y coordinates of the bounding box
        # center:
        bounging_box_center_absolute_x_coord = (
            bounding_box['x'] + (bounding_box['width'] / 2)
        )
        bounging_box_center_absolute_y_coord = (
            bounding_box['y'] + (bounding_box['height'] / 2)
        )

        # getting the required information about the grid cell that contains
        # its center:
        (
            cell_row_index,
            cell_column_index,
            cell_corner_x_coord,
            cell_corner_y_coord
        ) = get_cell_containing_bounding_box_center(
            center_absolute_x_and_y_coords=(
                bounging_box_center_absolute_x_coord,
                bounging_box_center_absolute_y_coord
            )
        )

        # transforming and normalizing the bounding box coordinates, now
        # meaning respectively for x and y the relative center offset from the
        # encoling cell top-left corner normalized to the grid cell size and
        # for width and height the relative heights and heights with respect
        # to the image sides:
        relative_x_coord = (
            (bounging_box_center_absolute_x_coord - cell_corner_x_coord) /
            OUTPUT_GRID_CELL_N_COLUMNS
        )
        relative_y_coord = (
            (bounging_box_center_absolute_y_coord - cell_corner_y_coord) /
            OUTPUT_GRID_CELL_N_ROWS
        )
        relative_width = bounding_box['width'] / IMAGE_N_COLUMNS
        relative_height = bounding_box['height'] / IMAGE_N_ROWS

        # getting the index of the anchor with closest aspect ratio to the
        # considered bounding box:
        label_anchor_index = get_index_of_anchor_with_closest_aspect_ratio(
            absolute_width=bounding_box['width'],
            absolute_height=bounding_box['height']
        )

        # associating the bounding box attributes to the respective anchor
        # labels - after checking there are no intrinsic limitations of the
        # employed design choices:
        label_cannot_be_associated_to_respective_anchor = (
            labels[cell_row_index, cell_column_index, label_anchor_index, :] !=
            [.0] * N_OUTPUTS_PER_ANCHOR
        ).any()
        if label_cannot_be_associated_to_respective_anchor:
            raise Exception(
                f"Either more than {N_ANCHORS_PER_CELL} anchors or a " +
                "better output resolution are required, as more bounding " +
                "boxes than the set number of anchors are falling within " +
                "the same output cell in this sample."
            )
        labels[cell_row_index, cell_column_index, label_anchor_index, :] = [
            1.0,  # FIXME: supposed to be an objectiveness score or an IoU?
            relative_x_coord,
            relative_y_coord,
            relative_width,
            relative_height
        ]

    return labels


(
    IMAGE_PATHS_TO_BOUNDING_BOXES,
    IMAGE_PATHS_TO_MODEL_OUTPUTS
) = load_labels_as_paths_to_bounding_boxes_and_model_outputs_dicts()

N_TRAINING_PLUS_VALIDATION_SAMPLES = len(IMAGE_PATHS_TO_BOUNDING_BOXES)


if __name__ == '__main__':
    if SHOW_BOUNDING_BOXES_STATISTICS:
        inspect_bounding_boxes_statistics_on_training_n_validation_set()

    samples_n_bounding_boxes_dataset = dataset_of_samples_and_bounding_boxes()

    if SHOW_DATASET_MOVIES:
        show_dataset_as_movie(
            ordered_samples_and_labels=samples_n_bounding_boxes_dataset,
            bounding_boxes_or_model_outputs='bounding_boxes'
        )

    samples_n_model_outputs_dataset = dataset_of_samples_and_model_outputs(
        # not shuffling when needing adjacent frames for showing the movie:
        shuffle=(not SHOW_DATASET_MOVIES)
    )

    if SHOW_DATASET_MOVIES:
        show_dataset_as_movie(
            ordered_samples_and_labels=samples_n_model_outputs_dataset,
            bounding_boxes_or_model_outputs='model_outputs'
        )

    (
        training_dataset, validation_dataset
    ) = split_dataset_into_batched_training_and_validation_sets(
        training_plus_validation_set=samples_n_model_outputs_dataset
    )
