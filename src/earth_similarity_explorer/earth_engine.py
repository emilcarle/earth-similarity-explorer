"""The export of the embeddings as int8 from Earth Engine via Google Drive, and their decoding.

Run the export stage from the project root with
uv run python -m earth_similarity_explorer.earth_engine <area id>
"""

import argparse
import shutil
import time
from pathlib import Path
from typing import Any

import ee
import numpy as np
import rasterio
from google.auth.transport.requests import AuthorizedSession
from requests import Response
from shapely import MultiPolygon, Polygon
from shapely.geometry import mapping

from earth_similarity_explorer.boundary import (
    define_tiles,
    define_utm_epsg,
    load_boundary,
)
from earth_similarity_explorer.config import AreaConfig, load_area_config
from earth_similarity_explorer.params import load_params
from earth_similarity_explorer.settings import (
    AREAS_DIR,
    DRIVE_EXPORT_FOLDER,
    DRIVE_FILES_URL,
    EMBEDDING_COLLECTION_ID,
    EMBEDDING_PIXEL_SIZE_M,
    EMBEDDINGS_DIR,
    EXPORT_GEOTIFF_OPTIONS,
    EXPORT_INT8_NODATA,
    EXPORT_INT8_SCALE,
    Settings,
)


# EARTH ENGINE
def initialize_earth_engine() -> None:
    """Connect to Earth Engine with the Cloud project."""
    ee.Initialize(project=Settings().ee_project)


def encode_int8(embeddings: ee.Image) -> ee.Image:
    """Return AlphaEarth's integer q of every embedding value as int8.

    AlphaEarth decodes q as v = sign(q)·(q / 127.5)², so q = round(sign(v)·√|v|·127.5).
    Masked pixels, including those outside a clip geometry, become EXPORT_INT8_NODATA.
    """
    # Invert the decoding by the square root of |v|, with the sign of v (scaled by 127.5)
    integers: ee.Image = (
        embeddings.abs()
        .sqrt()
        .multiply(image2=embeddings.signum())
        .multiply(image2=EXPORT_INT8_SCALE)
        .round()
    )
    # Fill the masked pixels with the no-data value (outside q's range of ±127)
    # Without sameFootprint=False the pixels outside a clip geometry would stay masked
    return integers.unmask(value=EXPORT_INT8_NODATA, sameFootprint=False).toInt8()


def build_embedding_image(region: ee.Geometry, year: int) -> ee.Image:
    """Return the embeddings of one year inside the region as int8 (see encode_int8)."""
    embeddings: ee.Image = (
        ee.ImageCollection(args=EMBEDDING_COLLECTION_ID)
        .filterBounds(geometry=region)
        .filterDate(start=f"{year}-01-01", end=f"{year + 1}-01-01")
        # Create a single mosaic of all the selected images
        .mosaic()
        # Mask the pixels outside the area
        .clip(clip_geometry=region)
    )
    return encode_int8(embeddings=embeddings)


def wait_for_task(task: Any, label: str) -> dict[str, Any]:
    """Wait for a started Earth Engine task, print its progress and return its metadata.

    A line is printed when the state, the current stage or the whole percentage
    changes, and at least every 30 seconds while the task runs.

    Raises:
        RuntimeError: the task failed or was cancelled.
    """
    # The task ID finds the task in the Earth Engine Code Editor's task list.
    print(f"Earth Engine task {label}: {task.id} started", flush=True)

    started: float = time.monotonic()
    last_state: str = ""
    last_print: float = started
    while True:
        # The full operation record has progress, stages and compute use
        operation: dict[str, Any] = ee.data.getOperation(operation_name=task.name)
        metadata: dict[str, Any] = operation["metadata"]
        # The current stage is the first one with work left
        stage: str = next(
            (
                stage["displayName"]
                for stage in metadata.get("stages", [])
                if int(stage.get("completeWorkUnits", 0)) < int(stage["totalWorkUnits"])
            ),
            "",
        )
        state: str = f"{metadata['state']} {metadata.get('progress', 0):.0%} {stage}"
        # Print on every change or after 30 s without a line to show a long task is alive
        if state != last_state or time.monotonic() - last_print >= 30:
            minutes, seconds = divmod(int(time.monotonic() - started), 60)
            eecu_hours: float = metadata.get("batchEecuUsageSeconds", 0) / 3600
            print(
                f"  {minutes:3d}:{seconds:02d}  {state.strip()}, "
                f"{eecu_hours:.3f} EECU-hours",
                flush=True,
            )
            last_state = state
            last_print = time.monotonic()
        if operation.get("done"):
            break
        time.sleep(10)

    # Failed and cancelled operations are also done, and carry Earth Engine's message
    if "error" in operation:
        raise RuntimeError(
            f"Earth Engine task {label}: {operation['error'].get('message')}"
        )
    return metadata


# GOOGLE DRIVE
def download_from_drive(
    folder_id: str, file_prefix: str, destination_dir: Path
) -> list[Path]:
    """Download files in a Drive folder whose names start with a prefix.

    Each file is deleted from Drive (permanently) once it is downloaded.
    Earth Engine splits large exports into several files with the same prefix, so a
    list is returned.

    Raises:
        FileNotFoundError: no file in the folder matches the prefix.
        requests.HTTPError: a Drive request fails.
    """
    session: AuthorizedSession = AuthorizedSession(
        credentials=ee.data.get_persistent_credentials()
    )

    # Drive's "contains" matches name prefixes word by word. startswith makes it exact
    listing: Response = session.get(
        url=DRIVE_FILES_URL,
        params={
            "q": f"'{folder_id}' in parents and name contains '{file_prefix}' "
            "and trashed = false",
            "fields": "files(id, name)",
            "pageSize": 1000,
        },
    )
    listing.raise_for_status()
    files: list[dict[str, Any]] = [
        file for file in listing.json()["files"] if file["name"].startswith(file_prefix)
    ]
    if not files:
        raise FileNotFoundError(f"no file {file_prefix}* in Drive folder {folder_id}")

    destination_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for file in files:
        path: Path = destination_dir / file["name"]
        # Streamed in 1 MB chunks, so a large file never sits in memory as a whole
        with session.get(
            url=f"{DRIVE_FILES_URL}/{file['id']}", params={"alt": "media"}, stream=True
        ) as download:
            download.raise_for_status()
            with path.open(mode="wb") as local_file:
                for chunk in download.iter_content(chunk_size=2**20):
                    local_file.write(chunk)
        # Delete only after a complete download, so a failure never loses the export
        session.delete(url=f"{DRIVE_FILES_URL}/{file['id']}").raise_for_status()
        print(f"Downloaded {path} ({path.stat().st_size / 1e6:.1f} MB)", flush=True)
        paths.append(path)
    return paths


# EXPORT AND DOWNLOAD
def export_embeddings(
    image: ee.Image,
    crs: str,
    west: int,
    north: int,
    size_m: int,
    file_prefix: str,
    destination_dir: Path,
) -> list[Path]:
    """Export a square of int8 embeddings as GeoTIFF via Google Drive and download it.

    The square has its north-west corner at (west, north) in the crs (a UTM zone, on
    the 10 m pixel grid) and sides of size_m. Blocks until the files are downloaded
    and deleted from Drive.

    Raises:
        RuntimeError: the Earth Engine task failed or was cancelled.
    """
    task: Any = ee.batch.Export.image.toDrive(
        image=image,
        description=file_prefix,
        folder=DRIVE_EXPORT_FOLDER,
        fileNamePrefix=file_prefix,
        crs=crs,
        # A pixel window instead of a region. Earth Engine writes the pixels outside a
        # region as 0 (a valid q), whatever the image holds there
        crsTransform=[
            EMBEDDING_PIXEL_SIZE_M,
            0,
            west,
            0,
            -EMBEDDING_PIXEL_SIZE_M,
            north,
        ],
        dimensions=f"{size_m // EMBEDDING_PIXEL_SIZE_M}x{size_m // EMBEDDING_PIXEL_SIZE_M}",
        fileFormat="GeoTIFF",
        # Record the no-data value in the GeoTIFF, so GIS tools and rasterio know it
        formatOptions={"noData": EXPORT_INT8_NODATA},
    )
    task.start()
    metadata: dict[str, Any] = wait_for_task(task=task, label=file_prefix)

    # The finished task links the Drive folder it wrote to as .../#folders/<id>
    folder_id: str = metadata["destinationUris"][0].rsplit(sep="/", maxsplit=1)[-1]
    return download_from_drive(
        folder_id=folder_id, file_prefix=file_prefix, destination_dir=destination_dir
    )


def compress_geotiff(path: Path) -> None:
    """Rewrite a downloaded GeoTIFF in place with the project's lossless compression."""
    # Read pixels, profile and band names before the file is overwritten
    with rasterio.open(fp=path) as source:
        stored: np.ndarray = source.read()
        profile: dict[str, Any] = source.profile
        band_names: tuple[str | None, ...] = source.descriptions
    with rasterio.open(
        fp=path, mode="w", **(profile | EXPORT_GEOTIFF_OPTIONS)
    ) as destination:
        destination.write(stored)
        destination.descriptions = band_names


# INT8 DECODING
def decode_int8(stored: np.ndarray) -> np.ndarray:
    """Return the embedding values of int8 integers as float32, with NaN for no-data."""
    # Decode by sign(q)·(q / 127.5)² in float64, which rounds to AlphaEarth's float32 values
    values: np.ndarray = (np.sign(stored) * (stored / EXPORT_INT8_SCALE) ** 2).astype(
        dtype=np.float32
    )
    # Mark the pixels outside the area as NaN
    values[stored == EXPORT_INT8_NODATA] = np.nan
    return values


# EXPORT STAGE
def main() -> None:
    """Export the embeddings of the area named on the command line, tile by tile.

    Tiles finished in an earlier run are skipped, so a failed run resumes and a rerun
    exports nothing. Delete the area's folder to export it again.
    """
    # Read the area id from the command line
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Export the int8 embeddings of one area from Earth Engine."
    )
    parser.add_argument("area_id", help="id of an area config in areas/")
    parser.add_argument(
        "--max-tiles",
        type=int,
        help="export at most this many new tiles in this run (default all)",
    )
    args: argparse.Namespace = parser.parse_args()
    area_id: str = args.area_id
    max_tiles: int | None = args.max_tiles

    # Load the area config and its boundary, find its UTM zone and its tiles
    config: AreaConfig = load_area_config(
        config_yaml_path=AREAS_DIR / f"{area_id}.yaml"
    )
    boundary: Polygon | MultiPolygon = load_boundary(path=config.boundary)
    crs: str = define_utm_epsg(boundary=boundary)
    tile_size_m: int = load_params().export.tile_size_m
    tiles: list[tuple[int, int]] = define_tiles(
        boundary=boundary, crs=crs, tile_size_m=tile_size_m
    )

    # One image for all tiles, with no-data outside the area
    initialize_earth_engine()
    image: ee.Image = build_embedding_image(
        region=ee.Geometry(geo_json=mapping(ob=boundary)), year=config.year
    )
    area_dir: Path = EMBEDDINGS_DIR / area_id
    exported: int = 0
    for number, (west, north) in enumerate(iterable=tiles, start=1):
        # Name each tile by its north-west corner in UTM metres
        tile_id: str = f"{area_id}_{west}_{north}"
        tile_dir: Path = area_dir / tile_id
        # Skip a tile finished in an earlier run
        if tile_dir.exists():
            print(f"Tile {number} of {len(tiles)}: {tile_id} exists", flush=True)
            continue
        # Stop once this run has exported as many tiles as asked for
        if exported == max_tiles:
            print(f"Stopped after {exported} new tiles (--max-tiles)", flush=True)
            break
        print(f"Tile {number} of {len(tiles)}: {tile_id}", flush=True)
        # Download into a temporary folder that becomes the tile's folder only once
        # every file is there, after removing what a failed run left behind
        partial_dir: Path = area_dir / f"{tile_id}.partial"
        shutil.rmtree(path=partial_dir, ignore_errors=True)
        paths: list[Path] = export_embeddings(
            image=image,
            crs=crs,
            west=west,
            north=north,
            size_m=tile_size_m,
            file_prefix=tile_id,
            destination_dir=partial_dir,
        )
        # Compress each file before the tile counts as finished
        for path in paths:
            compress_geotiff(path=path)
        partial_dir.rename(target=tile_dir)
        exported += 1


if __name__ == "__main__":
    main()
