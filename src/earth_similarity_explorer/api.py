"""Search API that scores every segment of an area against a clicked segment.

Run from the project root with uv run uvicorn earth_similarity_explorer.api:app
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Response, status
from pydantic import BaseModel

from earth_similarity_explorer.cloud import PCA, build_cloud, fit_pca
from earth_similarity_explorer.params import load_params
from earth_similarity_explorer.search import (
    encode_scores,
    load_vectors,
    score_similarity,
)
from earth_similarity_explorer.serving import build_id_raster, encode_id_png
from earth_similarity_explorer.settings import (
    CLOUD_MIN_SEGMENTS,
    GRID_FILE_SUFFIX,
    VECTORS_DIR,
)


# REQUEST MODELS
class BoundingBox(BaseModel):
    """A rectangle in the area's UTM coordinates (metres), from the query parameters."""

    west: float
    south: float
    east: float
    north: float


# RESPONSE MODELS
class Health(BaseModel):
    """The API's status and the number of segments of each loaded area."""

    status: str
    areas: dict[str, int]


class AreaMetadata(BaseModel):
    """Where the area's ID raster lies in UTM metres and how many segments it has."""

    west: float
    north: float
    pixel_size_m: int
    width: int
    height: int
    segment_count: int


# LOADING
@dataclass(frozen=True)
class LoadedArea:
    """Everything the API keeps in memory for one area."""

    vectors: np.ndarray
    # Segment centres in UTM metres and pixel counts, one per segment in segment_id order
    x: np.ndarray
    y: np.ndarray
    weights: np.ndarray
    # Axes of the whole area's PCA, which keep the clouds of selections from mirroring
    global_axes: np.ndarray
    metadata: AreaMetadata
    id_png: bytes


def load_area(path: Path, pixel_size_m: int) -> LoadedArea:
    """Read an area's vector table, fit its global PCA and build its ID raster as a PNG."""
    # Load the int8 vectors for the search and the cloud
    vectors: np.ndarray = load_vectors(path=path)
    segments: pd.DataFrame = pd.read_parquet(
        path=path, columns=["x", "y", "pixel_count"]
    )
    x: np.ndarray = segments["x"].to_numpy()
    y: np.ndarray = segments["y"].to_numpy()
    weights: np.ndarray = segments["pixel_count"].to_numpy()

    # Fit the global PCA once (with the largest-component sign rule)
    global_pca: PCA = fit_pca(vectors=vectors, weights=weights)

    # The segment centres give the ID raster that tells the page where each segment lies.
    ids, (west, north) = build_id_raster(x=x, y=y, pixel_size_m=pixel_size_m)
    metadata: AreaMetadata = AreaMetadata(
        west=west,
        north=north,
        pixel_size_m=pixel_size_m,
        width=ids.shape[1],
        height=ids.shape[0],
        segment_count=len(vectors),
    )
    return LoadedArea(
        vectors=vectors,
        x=x,
        y=y,
        weights=weights,
        global_axes=global_pca.axes,
        metadata=metadata,
        id_png=encode_id_png(ids=ids),
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    """Load every area's grid segmentation once, before the first request."""
    # Each grid cell is one pixel of the ID raster
    cell_size_m: int = load_params().grid.cell_size_m
    # One entry per area, keyed by the area id in the file name
    areas: dict[str, LoadedArea] = {
        path.stem.removesuffix(GRID_FILE_SUFFIX): load_area(
            path=path, pixel_size_m=cell_size_m
        )
        for path in sorted(VECTORS_DIR.glob(pattern=f"*{GRID_FILE_SUFFIX}.parquet"))
    }
    app.state.areas = areas
    # The API answers requests while the lifespan waits
    yield


app: FastAPI = FastAPI(title="Earth Similarity Explorer", lifespan=lifespan)


def get_area(area: str) -> LoadedArea:
    """Return a loaded area, or answer 404 when the API has no area of that name."""
    loaded: LoadedArea | None = app.state.areas.get(area)
    if loaded is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Unknown area {area}"
        )
    return loaded


# ENDPOINTS
@app.get(path="/health")
def health() -> Health:
    """Report that the API runs and how many segments each loaded area has."""
    return Health(
        status="ok",
        areas={
            area: loaded.metadata.segment_count
            for area, loaded in app.state.areas.items()
        },
    )


@app.get(path="/areas/{area}")
def area_metadata(area: str) -> AreaMetadata:
    """Return where the area's ID raster lies and how many segments it has."""
    return get_area(area=area).metadata


@app.get(path="/areas/{area}/segment_ids.png")
def segment_ids(area: str) -> Response:
    """Return the area's ID raster as an RGB PNG with one 24-bit ID per pixel."""
    return Response(content=get_area(area=area).id_png, media_type="image/png")


@app.get(path="/areas/{area}/pixel_counts")
def pixel_counts(area: str) -> Response:
    """Return each segment's pixel count (uint32, little-endian) in segment_id order.

    The page draws one cloud point per pixel. With the 100 m ID raster it doesn't see
    the pixels themselves, so it needs their number per segment.
    """
    weights: np.ndarray = get_area(area=area).weights
    return Response(
        content=weights.astype(dtype="<u4").tobytes(),
        media_type="application/octet-stream",
    )


@app.get(path="/areas/{area}/similar")
def similar(area: str, segment: Annotated[int, Query(ge=0)]) -> Response:
    """Return the similarity of every segment to the queried one, one byte each."""
    # Check that the segment exists in the area
    vectors: np.ndarray = get_area(area=area).vectors
    if segment >= len(vectors):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Area {area} has no segment {segment}",
        )

    # Score every segment and send the scores as raw bytes in segment_id order
    scores: np.ndarray = score_similarity(vectors=vectors, query_id=segment)
    return Response(
        content=encode_scores(scores=scores), media_type="application/octet-stream"
    )


@app.get(path="/areas/{area}/cloud")
def cloud(area: str, bbox: Annotated[BoundingBox, Query()]) -> Response:
    """Return the 3D cloud of the segments whose centre lies in the bbox.

    The body holds the sorted segment IDs (uint32, little-endian) followed by one
    position per segment (3 × uint8). A header gives the explained variance of the
    three axes.
    """
    # Select the segments whose centre lies inside the bbox (sorted by segment_id)
    loaded: LoadedArea = get_area(area=area)
    segment_ids: np.ndarray = np.flatnonzero(
        (loaded.x >= bbox.west)
        & (loaded.x <= bbox.east)
        & (loaded.y >= bbox.south)
        & (loaded.y <= bbox.north)
    )
    # Refuse selections too small for a stable PCA
    if segment_ids.size < CLOUD_MIN_SEGMENTS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"Select at least {CLOUD_MIN_SEGMENTS} segments for a cloud "
            f"(this selection has {segment_ids.size})",
        )
    # Fit the selection's PCA with the global axes as the sign reference
    positions, pca = build_cloud(
        vectors=loaded.vectors[segment_ids],
        weights=loaded.weights[segment_ids],
        reference_axes=loaded.global_axes,
    )
    return Response(
        content=segment_ids.astype(dtype="<u4").tobytes() + positions.tobytes(),
        media_type="application/octet-stream",
        headers={
            "X-Explained-Variance": ",".join(
                f"{share:.3f}" for share in pca.explained_variance
            )
        },
    )
