"""
Search API that scores every segment of an area against a clicked segment.
Run from the project root with uv run uvicorn earth_similarity_explorer.api:app
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Annotated

import numpy as np
from fastapi import FastAPI, HTTPException, Query, Response, status
from pydantic import BaseModel

from earth_similarity_explorer.search import (
    encode_scores,
    load_vectors,
    score_similarity,
)
from earth_similarity_explorer.settings import VECTORS_DIR


class Health(BaseModel):
    """The API's status and the number of segments of each loaded area."""

    status: str
    areas: dict[str, int]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    """Load the grid vectors of every area once, before the first request."""
    # One int8 matrix per area, keyed by the area id in the file name
    app.state.vectors = {
        path.stem.removesuffix("_grid"): load_vectors(path=path)
        for path in sorted(VECTORS_DIR.glob(pattern="*_grid.parquet"))
    }
    # The API answers requests while the lifespan waits here
    yield


app: FastAPI = FastAPI(title="Earth Similarity Explorer", lifespan=lifespan)


# ENDPOINTS
@app.get(path="/health")
def health() -> Health:
    """Report that the API runs and how many segments each loaded area has."""
    return Health(
        status="ok",
        areas={area: len(vectors) for area, vectors in app.state.vectors.items()},
    )


@app.get(path="/areas/{area}/similar")
def similar(area: str, segment: Annotated[int, Query(ge=0)]) -> Response:
    """Return the similarity of every segment to the queried one, one byte each."""
    # Look up the area's vectors and check that the segment exists in it
    vectors: np.ndarray | None = app.state.vectors.get(area)
    if vectors is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Unknown area {area}"
        )
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
