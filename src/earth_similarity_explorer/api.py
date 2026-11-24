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

from earth_similarity_explorer.params import load_params
from earth_similarity_explorer.search import (
    encode_scores,
    load_vectors,
    score_similarity,
)
from earth_similarity_explorer.serving import build_id_raster, encode_id_png
from earth_similarity_explorer.settings import GRID_FILE_SUFFIX, VECTORS_DIR


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
    metadata: AreaMetadata
    id_png: bytes


def load_area(path: Path, pixel_size_m: int) -> LoadedArea:
    """Read an area's vector table and build its ID raster as a PNG."""
    # Load the int8 vectors for the search
    vectors: np.ndarray = load_vectors(path=path)

    # The segment centres give the ID raster that tells the page where each segment lies.
    centres: pd.DataFrame = pd.read_parquet(path=path, columns=["x", "y"])
    ids, (west, north) = build_id_raster(
        x=centres["x"].to_numpy(), y=centres["y"].to_numpy(), pixel_size_m=pixel_size_m
    )
    metadata: AreaMetadata = AreaMetadata(
        west=west,
        north=north,
        pixel_size_m=pixel_size_m,
        width=ids.shape[1],
        height=ids.shape[0],
        segment_count=len(vectors),
    )
    return LoadedArea(vectors=vectors, metadata=metadata, id_png=encode_id_png(ids=ids))


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
