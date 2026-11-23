"""
Create quereable vector tables for areas based on input .tif files (clustered in Earth Engine).
Run from the project root: uv run python -m earth_similarity_explorer.vectors <area id>
"""

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio
from pandas.core.frame import DataFrame

from earth_similarity_explorer.settings import (
    EMBEDDING_BANDS,
    EMBEDDING_INT8_SCALE,
    GRID_SEGMENTATION_DIR,
    VECTORS_DIR,
)


def build_vector_table(
    means: np.ndarray, pixel_counts: np.ndarray, x: np.ndarray, y: np.ndarray
) -> pd.DataFrame:
    """Return one row per unit with its centre, pixel count, homogeneity and vector.

    Means holds each unit's raw mean embedding (N × 64) and x, y (its centroid in UTM metres).
    Rows are sorted north to south, then west to east.
    """
    # The length of the raw mean: 1 when all pixels are alike, lower when mixed.
    # keepdims returns it as a column (N × 1), so each row is divided by its own length
    homogeneity: np.ndarray = np.linalg.norm(x=means, axis=1, keepdims=True)
    # Set the vector to unit length, so a dot product of two vectors is their cosine similarity
    vectors: np.ndarray = means / homogeneity
    # Convert to Int8 (compression)
    stored: np.ndarray = np.round(a=vectors * EMBEDDING_INT8_SCALE).astype(
        dtype=np.int8
    )

    table: pd.DataFrame = pd.DataFrame(
        data={
            "x": x,
            "y": y,
            "pixel_count": pixel_counts.astype(dtype=np.int32),
            "homogeneity": homogeneity[:, 0].astype(dtype=np.float32),
            **{band: stored[:, i] for i, band in enumerate(iterable=EMBEDDING_BANDS)},
        }
    )
    # North to south, then west to east. A stable sort keeps ties in order.
    table: DataFrame = table.sort_values(
        by=["y", "x"], ascending=[False, True], kind="stable", ignore_index=True
    )
    # Add unit_id column based on the order.
    table.insert(loc=0, column="unit_id", value=np.arange(len(table), dtype=np.int32))
    return table


# READERS: one per unit kind, each returning the inputs for build_vector_table
def read_grid_segmentation(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read the grid GeoTIFF and return each cell's raw mean, pixel count and centroid.
    Cells without any pixel inside the area are masked (NaN) in the GeoTIFF and left out.
    """
    # Read all bands, then pick the means and the count by the names.
    with rasterio.open(fp=path) as dataset:
        bands: np.ndarray = dataset.read()
        names: list[str] = list(dataset.descriptions)
        transform: Any = dataset.transform
    means_raster: np.ndarray = bands[[names.index(band) for band in EMBEDDING_BANDS]]
    count_raster: np.ndarray = bands[names.index("pixel_count")]

    # Mask cells inside the area: their pixel count is not NaN
    inside: np.ndarray = ~np.isnan(count_raster)
    # Get values of the pixels inside the mask (aka. unmasked)
    pixel_counts: np.ndarray = count_raster[inside]
    means: np.ndarray = means_raster[:, inside].T
    # Get the centroid of each cell, from its row and column, in UTM metres
    rows, cols = np.nonzero(inside)
    x, y = rasterio.transform.xy(
        transform=transform, rows=rows, cols=cols, offset="center"
    )
    return means, pixel_counts, x, y


def main() -> None:
    """Write the vectors table based on the provided area named on the command line."""
    # Read the area id from the command line
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Build the vectors table of one area's grid."
    )
    parser.add_argument("area_id", help="id of an area config in areas/")
    area_id: str = parser.parse_args().area_id

    # Read the grid cells and turn them into one table row each
    means, pixel_counts, x, y = read_grid_segmentation(
        path=GRID_SEGMENTATION_DIR / f"{area_id}_grid.tif"
    )
    table: pd.DataFrame = build_vector_table(
        means=means, pixel_counts=pixel_counts, x=x, y=y
    )

    # Write the table as Parquet, without pandas' index: unit_id already numbers the rows
    VECTORS_DIR.mkdir(parents=True, exist_ok=True)
    path: Path = VECTORS_DIR / f"{area_id}_grid.parquet"
    table.to_parquet(path=path, index=False)
    print(f"Wrote {len(table)} units to {path}", flush=True)


if __name__ == "__main__":
    main()
