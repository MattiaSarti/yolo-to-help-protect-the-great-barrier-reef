"""
Convenient definitions of common constants.
"""


from typing import Tuple

from numpy import arange, meshgrid, ndarray, stack


def compute_grid_cell_centers_xy_coords() -> Tuple[ndarray, ndarray]:
    """
    Return a 2D array representing the output grid cell centers'
    (x, y) coordinates.  # FIXME
    ---
        Outputs' Shapes:
            - (OUTPUT_GRID_N_ROWS, OUTPUT_GRID_N_COLUMNS, 2)  # FIXME
            - (OUTPUT_GRID_N_ROWS*OUTPUT_GRID_N_COLUMNS, 2)
            - (OUTPUT_GRID_N_ROWS, OUTPUT_GRID_N_COLUMNS, 2)  # FIXME
        Outputs' Meanings:
            - the first dimension is the row index of the grid cell and the
            second dimension is the column index of the grid cell, while the
            third dimension represents the tuple of center coordinates (x, y)
            of the considered grid cell  # FIXME
            - the first dimension index ranges from low x to high x first and
            from low y to high y then, as a snake, and it ranges over the the
            differen cells, while the second dimension represents the tuple of
            center coordinates (x, y) of the considered grid cell
            -  # FIXME
        Output Shape:
        Output Meaning: 
    """
    centers_x_coords_values = arange(
        start=int(OUTPUT_GRID_CELL_N_COLUMNS / 2),
        stop=IMAGE_N_COLUMNS,
        step=OUTPUT_GRID_CELL_N_COLUMNS
    )
    assert centers_x_coords_values.shape == (OUTPUT_GRID_N_COLUMNS,)
    centers_y_coords_values = arange(
        start=int(OUTPUT_GRID_CELL_N_ROWS),
        stop=IMAGE_N_ROWS,
        step=OUTPUT_GRID_CELL_N_ROWS
    )
    assert centers_y_coords_values.shape == (OUTPUT_GRID_N_ROWS,)

    (
        centers_x_coords, centers_y_coords
    ) = meshgrid(centers_x_coords_values, centers_y_coords_values)

    centers_xy_coords = stack(
        arrays=(centers_x_coords, centers_y_coords),
        axis=-1
    )
    flattened_centers_xy_coords = centers_xy_coords.reshape((-1, 2))

    indexes_of_grid_cells_flattenings = None
    raise NotImplementedError

    return (
        centers_xy_coords,
        flattened_centers_xy_coords,
        indexes_of_grid_cells_flattenings,
    )  # FIXME: https://stackoverflow.com/questions/41841354/keeping-track-of-indices-change-in-numpy-reshape


DOWNSAMPLING_STEPS = 4

IMAGE_N_CHANNELS = 3
IMAGE_N_COLUMNS = 1280
IMAGE_N_ROWS = 720

N_CONVOLUTIONS_AT_SAME_RESOLUTION = 3
N_OUTPUTS_PER_ANCHOR = 5

OUTPUT_GRID_CELL_N_ANCHORS = 3
OUTPUT_GRID_CELL_N_COLUMNS = 16  # TODO
OUTPUT_GRID_CELL_N_ROWS = 16  # TODO

OUTPUT_GRID_N_COLUMNS = int(IMAGE_N_COLUMNS / OUTPUT_GRID_CELL_N_COLUMNS)
OUTPUT_GRID_N_ROWS = int(IMAGE_N_ROWS / OUTPUT_GRID_CELL_N_ROWS)

(
    UNFLATTENED_OUTPUT_GRID_CELL_CENTERS_XY_COORDS,
    FLATTENED_OUTPUT_GRID_CELL_CENTERS_XY_COORDS,
    FLATTENED_OUTPUT_GRID_CELL_CENTERS_ROW_AND_COLUMN_INDEXES
) = compute_grid_cell_centers_xy_coords()
