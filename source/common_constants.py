"""
Convenient definitions of common constants.
"""


from numpy import arange, meshgrid, ndarray, stack


def compute_grid_cell_centers_xy_coords() -> ndarray:
    """
    Return a 2D array representing the output grid cell centers'
    (x, y) coordinates.
    ---
        Output Shape:
            (OUTPUT_GRID_N_ROWS*OUTPUT_GRID_N_COLUMNS, 2)
        Output Meaning: 
            the first dimension index ranges from low x to high x first and
            from low y to high y then, as a snake, and it ranges over the the
            differen cells, while the second dimension represents the tuple of
            center coordinates (x, y) of the considered grid cell
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
    raise Exception()  # FIXME: this it the top-left corner, while I need the center

    (
        centers_x_coords, centers_y_coords
    ) = meshgrid(centers_x_coords_values, centers_y_coords_values)

    return stack(
        arrays=(centers_x_coords, centers_y_coords),
        axis=-1
    ).reshape((-1, 2))


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

OUTPUT_GRID_CELL_CENTERS_XY_COORDS = compute_grid_cell_centers_xy_coords()
