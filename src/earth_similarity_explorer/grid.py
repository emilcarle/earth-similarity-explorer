"""Construction of grid segmentation based on mean embeddings (the segmentation baseline).
Run from the project root: uv run python -m earth_similarity_explorer.grid <area id>
"""

import argparse

import ee
from shapely import MultiPolygon, Polygon
from shapely.geometry import mapping

from earth_similarity_explorer.boundary import define_utm_epsg, load_boundary
from earth_similarity_explorer.config import AreaConfig, load_area_config
from earth_similarity_explorer.earth_engine import (
    export_and_download,
    initialize_earth_engine,
)
from earth_similarity_explorer.params import Params, load_params
from earth_similarity_explorer.settings import (
    AREAS_DIR,
    DRIVE_EXPORT_FOLDER,
    EMBEDDING_COLLECTION_ID,
    EMBEDDING_PIXEL_SIZE_M,
    GRID_SEGMENTATION_DIR,
)


def build_grid_image(
    region: ee.Geometry, year: int, crs: str, cell_size_m: int
) -> ee.Image:
    """Return the mean embedding and pixel count of every grid cell as float32.

    Bands: A00–A63, the mean of the pixels inside the region (not renormalized) +
    pixel_count. Cells outside the area are masked (NaN in the GeoTIFF).
    """
    pixels_per_cell: int = (cell_size_m // EMBEDDING_PIXEL_SIZE_M) ** 2

    embeddings: ee.Image = (
        ee.ImageCollection(args=EMBEDDING_COLLECTION_ID)
        .filterBounds(geometry=region)
        .filterDate(start=f"{year}-01-01", end=f"{year + 1}-01-01")
        # Create a single mosaic of all the selected tiles
        .mosaic()
        # mosaic() drops the projection so reduceResolution needs the 10 m input grid
        .setDefaultProjection(
            crs=crs,
            crsTransform=[EMBEDDING_PIXEL_SIZE_M, 0, 0, 0, -EMBEDDING_PIXEL_SIZE_M, 0],
        )
        # Mask the pixels outside the area before averaging, so edge cells use and
        # count only the pixels inside
        .clip(clip_geometry=region)
    )
    means: ee.Image = embeddings.reduceResolution(
        reducer=ee.Reducer.mean(), maxPixels=pixels_per_cell
    )
    # Creata a pixel count band
    pixel_count: ee.Image = (
        embeddings.select(selectors=["A00"])
        .reduceResolution(reducer=ee.Reducer.count(), maxPixels=pixels_per_cell)
        .rename(names=["pixel_count"])
    )
    # A GeoTIFF has one data type for all bands. float32 keeps the means lossless
    # and the counts exact up to 16.7 million
    return means.addBands(srcImg=pixel_count).toFloat()


def main() -> None:
    """Export the grid of the area named on the command line to the export folder."""
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Export the grid segmentation of mean embeddings for one area."
    )
    parser.add_argument("area_id", help="id of an area config in areas/")
    area_id: str = parser.parse_args().area_id

    params: Params = load_params()
    config: AreaConfig = load_area_config(
        config_yaml_path=AREAS_DIR / f"{area_id}.yaml"
    )
    boundary: Polygon | MultiPolygon = load_boundary(path=config.boundary)
    crs: str = define_utm_epsg(boundary=boundary)
    cell_size_m: int = params.grid.cell_size_m

    initialize_earth_engine()
    region: ee.Geometry = ee.Geometry(geo_json=mapping(ob=boundary))
    export_and_download(
        image=build_grid_image(
            region=region, year=config.year, crs=crs, cell_size_m=cell_size_m
        ),
        # A rectangle one cell wider than the area: the polygon itself would mask
        # whole edge cells, and with them their pixels inside. clip() alone decides
        region=region.buffer(distance=cell_size_m).bounds(),
        crs=crs,
        # Cells anchored at (0, 0), so their edges lie on the 10 m pixel grid
        crs_transform=[cell_size_m, 0, 0, 0, -cell_size_m, 0],
        drive_folder=DRIVE_EXPORT_FOLDER,
        file_prefix=f"{area_id}_grid",
        destination_dir=GRID_SEGMENTATION_DIR,
    )


if __name__ == "__main__":
    main()
