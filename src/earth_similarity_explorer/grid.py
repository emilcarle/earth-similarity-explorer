"""Grid segmentation of mean embeddings, used as a baseline comparison for the k-means clustering.

Run from the project root with uv run python -m earth_similarity_explorer.grid <area id>
"""

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.transform import from_origin

from earth_similarity_explorer.earth_engine import decode_int8
from earth_similarity_explorer.params import load_params
from earth_similarity_explorer.settings import (
    EMBEDDING_BANDS,
    EMBEDDING_PIXEL_SIZE_M,
    EMBEDDINGS_DIR,
    GRID_FILE_SUFFIX,
    GRID_SEGMENTATION_DIR,
    PIXEL_COUNT_BAND,
)


def create_grid_segmentation(
    embeddings: np.ndarray, transform: Any, cell_size_m: int
) -> tuple[np.ndarray, Any]:
    """Return the mean embedding and pixel count of every grid cell, and the grid's transform."""
    cell_pixels: int = cell_size_m // EMBEDDING_PIXEL_SIZE_M
    # Snap the raster's north-west corner (transform.c, transform.f) outward to the cell grid
    west: int = math.floor(transform.c / cell_size_m) * cell_size_m
    north: int = math.ceil(transform.f / cell_size_m) * cell_size_m
    # Pad with NaN so the raster starts at a cell edge and holds whole cells only
    left: int = int(transform.c - west) // EMBEDDING_PIXEL_SIZE_M
    top: int = int(north - transform.f) // EMBEDDING_PIXEL_SIZE_M
    bands, rows, cols = embeddings.shape
    padded: np.ndarray = np.pad(
        array=embeddings,
        pad_width=(
            (0, 0),
            (top, -(rows + top) % cell_pixels),
            (left, -(cols + left) % cell_pixels),
        ),
        constant_values=np.nan,
    )
    # Reshape into blocks (bands × cell rows × pixels × cell cols × pixels)
    blocks: np.ndarray = padded.reshape(
        bands,
        padded.shape[1] // cell_pixels,
        cell_pixels,
        padded.shape[2] // cell_pixels,
        cell_pixels,
    )
    # Count the pixels inside the area per cell, with NaN for cells without any
    counts: np.ndarray = (~np.isnan(blocks[0])).sum(axis=(1, 3))
    counts_or_nan: np.ndarray = np.where(counts > 0, counts, np.nan)
    # Calculate the mean of each band per cell over the pixels inside (sums in float64)
    means: np.ndarray = (
        np.nansum(a=blocks, axis=(2, 4), dtype=np.float64) / counts_or_nan
    )
    grid: np.ndarray = np.concatenate([means, counts_or_nan[None]]).astype(
        dtype=np.float32
    )
    return grid, from_origin(
        west=west, north=north, xsize=cell_size_m, ysize=cell_size_m
    )


def mosaic_tile_grids(
    tile_grids: list[tuple[np.ndarray, Any]], cell_size_m: int
) -> tuple[np.ndarray, Any]:
    """Return one grid that holds the cells of all tile grids, and its transform.

    Each tile grid comes with its transform, as create_grid_segmentation returns it.
    Cells that no tile covers are NaN.
    """
    # Calculate the extent of all tiles together (UTM metres)
    west: float = min(transform.c for _, transform in tile_grids)
    north: float = max(transform.f for _, transform in tile_grids)
    east: float = max(
        transform.c + tile_grid.shape[2] * cell_size_m
        for tile_grid, transform in tile_grids
    )
    south: float = min(
        transform.f - tile_grid.shape[1] * cell_size_m
        for tile_grid, transform in tile_grids
    )
    # Create an empty grid over the extent with the tiles' bands
    grid: np.ndarray = np.full(
        shape=(
            tile_grids[0][0].shape[0],
            round((north - south) / cell_size_m),
            round((east - west) / cell_size_m),
        ),
        fill_value=np.nan,
        dtype=np.float32,
    )
    # Place each tile's cells by the offset of its north-west corner (in cells)
    for tile_grid, transform in tile_grids:
        row: int = round((north - transform.f) / cell_size_m)
        col: int = round((transform.c - west) / cell_size_m)
        grid[:, row : row + tile_grid.shape[1], col : col + tile_grid.shape[2]] = (
            tile_grid
        )
    return grid, from_origin(
        west=west, north=north, xsize=cell_size_m, ysize=cell_size_m
    )


def write_grid(grid: np.ndarray, transform: Any, crs: Any, path: Path) -> None:
    """Write a grid as float32 GeoTIFF with the band names as descriptions."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        fp=path,
        mode="w",
        driver="GTiff",
        width=grid.shape[2],
        height=grid.shape[1],
        count=grid.shape[0],
        dtype="float32",
        crs=crs,
        transform=transform,
        nodata=np.nan,
    ) as destination:
        destination.write(grid)
        destination.descriptions = (*EMBEDDING_BANDS, PIXEL_COUNT_BAND)


def main() -> None:
    """Write the grid of the area named on the command line from its exported embeddings."""
    # Read the area id from the command line
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Build the grid segmentation of mean embeddings for one area."
    )
    parser.add_argument("area_id", help="id of an area config in areas/")
    area_id: str = parser.parse_args().area_id
    cell_size_m: int = load_params().grid.cell_size_m

    # Find the finished tiles (a .partial folder is a tile still downloading)
    tile_dirs: list[Path] = sorted(
        path
        for path in (EMBEDDINGS_DIR / area_id).iterdir()
        if path.is_dir() and path.suffix != ".partial"
    )
    with rasterio.open(fp=next(tile_dirs[0].glob(pattern="*.tif"))) as source:
        crs: Any = source.crs

    # Average every tile on its own, since whole cells fill a tile
    tile_grids: list[tuple[np.ndarray, Any]] = []
    for tile_dir in tile_dirs:
        # Read the tile as one raster (Earth Engine may split it into several files)
        stored, transform = merge(sources=sorted(tile_dir.glob(pattern="*.tif")))
        tile_grids.append(
            create_grid_segmentation(
                embeddings=decode_int8(stored=stored),
                transform=transform,
                cell_size_m=cell_size_m,
            )
        )

    # Join the tiles into one grid and write it
    grid, grid_transform = mosaic_tile_grids(
        tile_grids=tile_grids, cell_size_m=cell_size_m
    )
    path: Path = GRID_SEGMENTATION_DIR / f"{area_id}{GRID_FILE_SUFFIX}.tif"
    write_grid(grid=grid, transform=grid_transform, crs=crs, path=path)
    print(f"Wrote {grid.shape[2]} × {grid.shape[1]} cells to {path}", flush=True)


if __name__ == "__main__":
    main()
