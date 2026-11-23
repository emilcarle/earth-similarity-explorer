"""Tests for the vectors stage: the table every unit kind shares, and its readers."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin

from earth_similarity_explorer.settings import EMBEDDING_BANDS
from earth_similarity_explorer.vectors import build_vector_table, read_grid_segmentation


# HELPER FUNCTIONS
def make_table(
    first_two: list[tuple[float, float]], centres: list[tuple[float, float]]
) -> pd.DataFrame:
    """Build a table from units whose mean has only A00 and A01 set."""
    means: np.ndarray = np.zeros(shape=(len(first_two), len(EMBEDDING_BANDS)))
    means[:, :2] = first_two
    xy: np.ndarray = np.array(object=centres)
    return build_vector_table(
        means=means,
        pixel_counts=np.arange(1, stop=len(first_two) + 1),
        x=xy[:, 0],
        y=xy[:, 1],
    )


# TESTS
def test_table_has_the_shared_columns_and_types() -> None:
    """The table has the agreed columns, in order, with compact types."""
    table: pd.DataFrame = make_table(first_two=[(0.6, 0.8)], centres=[(50.0, 50.0)])

    assert list(table.columns) == [
        "unit_id",
        "x",
        "y",
        "pixel_count",
        "homogeneity",
        *EMBEDDING_BANDS,
    ]
    assert table["unit_id"].dtype == np.int32
    assert table["pixel_count"].dtype == np.int32
    assert table["homogeneity"].dtype == np.float32
    assert (table[EMBEDDING_BANDS].dtypes == np.int8).all()


def test_mean_becomes_homogeneity_and_int8_unit_vector() -> None:
    """A mean of length 0.9 gives homogeneity 0.9 and the vector [0.6, 0.8] × 127."""
    table: pd.DataFrame = make_table(first_two=[(0.54, 0.72)], centres=[(50.0, 50.0)])
    row: pd.Series = table.iloc[0]

    assert row["homogeneity"] == np.float32(0.9)
    # 0.6 × 127 = 76.2 and 0.8 × 127 = 101.6, rounded to the nearest integer
    assert (row["A00"], row["A01"]) == (76, 102)
    assert (row[EMBEDDING_BANDS[2:]] == 0).all()


def test_rows_run_north_to_south_then_west_to_east() -> None:
    """Rows are sorted by centre, and unit_id is the new row number."""
    # Given as south-east, north-west, north-east
    table: pd.DataFrame = make_table(
        first_two=[(1.0, 0.0)] * 3,
        centres=[(150.0, 50.0), (50.0, 150.0), (150.0, 150.0)],
    )

    assert table["unit_id"].tolist() == [0, 1, 2]
    assert table[["x", "y"]].values.tolist() == [
        [50.0, 150.0],
        [150.0, 150.0],
        [150.0, 50.0],
    ]
    # Each unit's other columns move with it: pixel counts 1, 2, 3 were given in input order
    assert table["pixel_count"].tolist() == [2, 3, 1]


def test_grid_reader_returns_cells_with_pixels_and_their_centres(
    tmp_path: Path,
) -> None:
    """Cells with pixels come back in row-major order, with centres, means and counts."""
    # A 2 × 2 grid of 100 m cells. The north-east cell lies outside the area (NaN)
    pixel_count: np.ndarray = np.array(object=[[100, np.nan], [37, 100]])
    means: np.ndarray = np.zeros(shape=(64, 2, 2))
    means[0] = [[0.1, np.nan], [0.3, 0.4]]
    path: Path = tmp_path / "grid.tif"
    with rasterio.open(
        fp=path,
        mode="w",
        driver="GTiff",
        height=2,
        width=2,
        count=65,
        dtype="float32",
        crs="EPSG:32632",
        transform=from_origin(west=670500, north=5283200, xsize=100, ysize=100),
    ) as dataset:
        # pixel_count first: the reader must find bands by name, not by position
        dataset.write(np.concatenate((pixel_count[np.newaxis], means)))
        dataset.descriptions = ("pixel_count", *EMBEDDING_BANDS)

    read_means, pixel_counts, x, y = read_grid_segmentation(path=path)

    assert pixel_counts.tolist() == [100, 37, 100]
    assert x.tolist() == [670550, 670550, 670650]
    assert y.tolist() == [5283150, 5283050, 5283050]
    assert read_means.shape == (3, 64)
    assert read_means[:, 0].tolist() == pytest.approx(expected=[0.1, 0.3, 0.4])
