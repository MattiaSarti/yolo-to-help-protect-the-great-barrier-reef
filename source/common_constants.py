"""
Convenient definitions of common constants.
"""


from typing import Tuple

from numpy import arange, meshgrid, ndarray, stack


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


DOWNSAMPLING_STEPS = 4

IMAGE_N_CHANNELS = 3
IMAGE_N_COLUMNS = 1280
IMAGE_N_ROWS = 720

N_OUTPUTS_PER_ANCHOR = 5

OUTPUT_GRID_CELL_N_ANCHORS = 3
OUTPUT_GRID_CELL_N_COLUMNS = 16  # NOTE: this may vary with the architecture
OUTPUT_GRID_CELL_N_ROWS = 16  # NOTE: this may vary with the architecture
# NOTE: common divisors of 1280 and 720: {1, 2, 4, 5, 8, 10, 16, 20, 40, 80},
# and the ones that respect the training-plus-validation set vounding boxes'
# distinction when using a single anchor are: {1, 2, 4, 5, 8, 10, 16}

OUTPUT_GRID_N_COLUMNS = int(IMAGE_N_COLUMNS / OUTPUT_GRID_CELL_N_COLUMNS)
OUTPUT_GRID_N_ROWS = int(IMAGE_N_ROWS / OUTPUT_GRID_CELL_N_ROWS)

(
    OUTPUT_GRID_CELL_CENTERS_XY_COORDS,
    OUTPUT_GRID_CELL_CORNERS_XY_COORDS
) = compute_grid_cell_centers_xy_coords()
