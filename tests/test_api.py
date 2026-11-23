"""Tests for the search API."""

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from httpx import Response

from earth_similarity_explorer import api
from earth_similarity_explorer.settings import EMBEDDING_BANDS
from earth_similarity_explorer.vectors import build_vector_table


# FIXTURES
@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """Start the API on one tiny area named test with three segments."""
    # Three segments in a row from west to east whose means point east, north and west
    means: np.ndarray = np.zeros(shape=(3, len(EMBEDDING_BANDS)))
    means[:, :2] = [[1, 0], [0, 1], [-1, 0]]
    table: pd.DataFrame = build_vector_table(
        means=means,
        pixel_counts=np.array(object=[100, 100, 100]),
        x=np.array(object=[50.0, 150.0, 250.0]),
        y=np.array(object=[50.0, 50.0, 50.0]),
    )
    table.to_parquet(path=tmp_path / "test_grid.parquet", index=False)

    # The lifespan loads the tables from tmp_path instead of data/vectors
    monkeypatch.setattr(target=api, name="VECTORS_DIR", value=tmp_path)
    # Entering the with block runs the lifespan, leaving it shuts the API down
    with TestClient(app=api.app) as test_client:
        yield test_client


# TESTS
def test_health_lists_each_loaded_area_with_its_segment_count(
    client: TestClient,
) -> None:
    """Health reports ok and every area found in the vectors folder."""
    response: Response = client.get(url="/health")

    assert response.json() == {"status": "ok", "areas": {"test": 3}}


def test_similar_returns_one_byte_per_segment_in_segment_id_order(
    client: TestClient,
) -> None:
    """A search returns raw bytes, one per segment, with the query itself at 255."""
    response: Response = client.get(url="/areas/test/similar", params={"segment": 0})

    assert response.headers["content-type"] == "application/octet-stream"
    # Cosines 1, 0 and −1 against segment 0 become bytes 255, 128 and 0
    assert list(response.content) == [255, 128, 0]


@pytest.mark.parametrize(
    argnames=("area", "segment", "status_code"),
    argvalues=[
        # An area without a vectors table
        ("nowhere", 0, 404),
        # One past the last segment
        ("test", 3, 404),
        # Negative IDs are rejected before the endpoint runs
        ("test", -1, 422),
    ],
)
def test_similar_rejects_unknown_areas_and_segments(
    client: TestClient, area: str, segment: int, status_code: int
) -> None:
    """Searches for an area or segment that doesn't exist return an error status."""
    response: Response = client.get(
        url=f"/areas/{area}/similar", params={"segment": segment}
    )

    assert response.status_code == status_code
