"""
Convenient definitions of common constants.
"""


from os import getcwd
from os.path import join as path_join
from typing import Tuple

from numpy import arange, meshgrid, ndarray, stack
# pylint: disable=import-error,no-name-in-module
from tensorflow import float32 as tf_float32, uint8 as tf_uint8
# pylint: enable=import-error,no-name-in-module


def compute_grid_cell_centers_xy_coords() -> Tuple[ndarray, ndarray]:
    """
    Return two 3D arrays respectively representing the output grid cell
    centers' (x, y) coordinates and top-left corners' (x, y) coordinates,
    indexed along the first two dimensions as rows and columns of cells in the
    output grid.
    ---
        Outputs' Shapes:
            - (OUTPUT_GRID_N_ROWS, OUTPUT_GRID_N_COLUMNS, 2)
            - (OUTPUT_GRID_N_ROWS, OUTPUT_GRID_N_COLUMNS, 2)
    ---
        Outputs' Meanings:
            - the first dimension is the row index of the grid cell and the
            second dimension is the column index of the grid cell, while the
            third dimension represents the tuple of center (x, y) coordinates
            of the considered grid cell
            - the first dimension is the row index of the grid cell and the
            second dimension is the column index of the grid cell, while the
            third dimension represents the tuple of top-left corner (x, y)
            coordinates of the considered grid cell
    """
    # x and y possible values spanned by grid cell centers:
    centers_x_coords_values = arange(
        start=int(OUTPUT_GRID_CELL_N_COLUMNS / 2),
        stop=IMAGE_N_COLUMNS,
        step=OUTPUT_GRID_CELL_N_COLUMNS
    )
    assert centers_x_coords_values.shape == (OUTPUT_GRID_N_COLUMNS,)
    centers_y_coords_values = arange(
        start=int(OUTPUT_GRID_CELL_N_ROWS / 2),
        stop=IMAGE_N_ROWS,
        step=OUTPUT_GRID_CELL_N_ROWS
    )
    assert centers_y_coords_values.shape == (OUTPUT_GRID_N_ROWS,)

    # x and y possible values spanned by grid cell top-left corners:
    corners_x_coords_values = arange(
        start=0,
        stop=IMAGE_N_COLUMNS,
        step=OUTPUT_GRID_CELL_N_COLUMNS
    )
    assert corners_x_coords_values.shape == (OUTPUT_GRID_N_COLUMNS,)
    corners_y_coords_values = arange(
        start=0,
        stop=IMAGE_N_ROWS,
        step=OUTPUT_GRID_CELL_N_ROWS
    )
    assert corners_y_coords_values.shape == (OUTPUT_GRID_N_ROWS,)

    # grid of cells containing the respective center x and y coordinates each:
    centers_xy_coords = stack(
        arrays=meshgrid(centers_x_coords_values, centers_y_coords_values),
        axis=-1
    )

    # grid of cells containing the respective top-left corner x and y
    # coordinates each:
    corners_xy_coords = stack(
        arrays=meshgrid(corners_x_coords_values, corners_y_coords_values),
        axis=-1
    )

    return (
        centers_xy_coords,
        corners_xy_coords
    )


def compute_weights_to_balance_anchors_emptiness() -> Tuple[float, float]:
    """
    Return the weights, for the loss function terms, that balance full vs
    empty anchors.
    """
    average_n_full_anchors_per_image = (
        AVERAGE_N_BOUNDING_BOXES_PER_IMAGE / N_ANCHORS_PER_IMAGE
    )
    average_n_empty_anchors_per_image = (
        N_ANCHORS_PER_IMAGE - average_n_full_anchors_per_image
    )

    full_anchors_weight = 1 / average_n_full_anchors_per_image
    empty_anchors_weight = 1 / average_n_empty_anchors_per_image

    weights_sum = full_anchors_weight + empty_anchors_weight

    normalized_full_anchors_weight = full_anchors_weight / weights_sum
    normalized_empty_anchors_weight = empty_anchors_weight / weights_sum

    return normalized_full_anchors_weight, normalized_empty_anchors_weight


AVERAGE_N_BOUNDING_BOXES_PER_IMAGE = 0.51

DATA_TYPE_FOR_INPUTS = tf_uint8
DATA_TYPE_FOR_OUTPUTS = tf_float32

IMAGE_N_CHANNELS = 3
IMAGE_N_COLUMNS = 1280
IMAGE_N_ROWS = 720

# MINIMUM_BOUNDING_BOX_HEIGHT = 13  # [pixels]
# MINIMUM_BOUNDING_BOX_WIDTH = 17  # [pixels]

N_OUTPUTS_PER_ANCHOR = 5

ANCHORS_WIDTH_VS_HEIGHT_WEIGHTS = (
    (0.6, 0.4),
    (0.5, 0.5),
    # (0.4, 0.6)  # NOTE: empirically observed: this anchor is less relevant
)
assert all(
    [
        (weight[0] + weight[1] == 1) for weight in
        ANCHORS_WIDTH_VS_HEIGHT_WEIGHTS
    ]
)
N_ANCHORS_PER_CELL = len(
    ANCHORS_WIDTH_VS_HEIGHT_WEIGHTS
)

OUTPUT_GRID_CELL_N_COLUMNS = 16  # NOTE: this may vary with the architecture
OUTPUT_GRID_CELL_N_ROWS = 16  # NOTE: this may vary with the architecture
# NOTE: common divisors of 1280 and 720: {1, 2, 4, 5, 8, 10, 16, 20, 40, 80},
# and the ones that respect the training-plus-validation set bounding boxes'
# distinction when using a single anchor are: {1, 2, 4, 5, 8, 10, 16}

OUTPUT_GRID_N_COLUMNS = int(IMAGE_N_COLUMNS / OUTPUT_GRID_CELL_N_COLUMNS)
OUTPUT_GRID_N_ROWS = int(IMAGE_N_ROWS / OUTPUT_GRID_CELL_N_ROWS)

N_ANCHORS_PER_IMAGE = (
    OUTPUT_GRID_N_COLUMNS * OUTPUT_GRID_N_ROWS * N_ANCHORS_PER_CELL
)

(
    OUTPUT_GRID_CELL_CENTERS_XY_COORDS,
    OUTPUT_GRID_CELL_CORNERS_XY_COORDS
) = compute_grid_cell_centers_xy_coords()

(
    LOSS_CONTRIBUTE_IMPORTANCE_OF_FULL_ANCHORS,
    LOSS_CONTRIBUTE_IMPORTANCE_OF_EMPTY_ANCHORS
) = compute_weights_to_balance_anchors_emptiness()
# FIXME: is this balancing reasonable?  with 0.999999990162037 vs
# 9.837962962962963e-09, using float32 will truncate the second term to 0!!
(
    LOSS_CONTRIBUTE_IMPORTANCE_OF_FULL_ANCHORS,
    LOSS_CONTRIBUTE_IMPORTANCE_OF_EMPTY_ANCHORS
) = (0.5, 0.5)

if __name__ != 'main_by_mattia':
    MODEL_PATH = path_join(
        getcwd(),
        'models',
        'model.h5'
    )
else:
    MODEL_PATH = path_join(getcwd(), 'model.h5')
