"""Tests for the grid segmentation built from the exported pixels."""

from typing import Any

import numpy as np
from rasterio.transform import from_origin

from earth_similarity_explorer.grid import create_grid_segmentation, mosaic_tile_grids


# TESTS
def test_cells_average_and_count_only_the_pixels_inside_the_area() -> None:
    """Cells anchored at (0, 0) average their pixels, skip NaN and are NaN when empty."""
    # One band of 2 × 5 pixels of 10 m with its corner at (10, 30), so neither edge
    # lies on the 20 m cell grid. NaN marks pixels outside the area.
    nan: float = np.nan
    embeddings: np.ndarray = np.array(
        object=[[[1, 2, 3, nan, nan], [5, nan, 7, nan, nan]]], dtype=np.float32
    )

    grid, transform = create_grid_segmentation(
        embeddings=embeddings,
        transform=from_origin(west=10, north=30, xsize=10, ysize=10),
        cell_size_m=20,
    )

    # The corner snaps outward to (0, 40), so the first cell row and column hold one
    # row or column of pixels each. The third cell column lies outside the area.
    expected_transform: Any = from_origin(west=0, north=40, xsize=20, ysize=20)
    assert transform == expected_transform
    np.testing.assert_array_equal(actual=grid[0], desired=[[1, 2.5, nan], [5, 7, nan]])
    np.testing.assert_array_equal(actual=grid[1], desired=[[1, 2, nan], [1, 1, nan]])


def test_mosaic_places_each_tile_at_its_corner() -> None:
    """Tile grids land at the offset of their corners, and uncovered cells stay NaN."""
    # Two tiles of 2 × 2 cells of 10 m, one in the north-west and one in the south-east
    north_west: np.ndarray = np.full(shape=(1, 2, 2), fill_value=1, dtype=np.float32)
    south_east: np.ndarray = np.full(shape=(1, 2, 2), fill_value=2, dtype=np.float32)

    grid, transform = mosaic_tile_grids(
        tile_grids=[
            (south_east, from_origin(west=20, north=20, xsize=10, ysize=10)),
            (north_west, from_origin(west=0, north=40, xsize=10, ysize=10)),
        ],
        cell_size_m=10,
    )

    nan: float = np.nan
    expected_transform: Any = from_origin(west=0, north=40, xsize=10, ysize=10)
    assert transform == expected_transform
    np.testing.assert_array_equal(
        actual=grid[0],
        desired=[
            [1, 1, nan, nan],
            [1, 1, nan, nan],
            [nan, nan, 2, 2],
            [nan, nan, 2, 2],
        ],
    )
