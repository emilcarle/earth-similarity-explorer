"""Tests for the search API."""

from collections.abc import Iterator
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from httpx import Response
from PIL import Image

from earth_similarity_explorer import api
from earth_similarity_explorer.settings import EMBEDDING_BANDS, GRID_FILE_SUFFIX
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
    table.to_parquet(path=tmp_path / f"test{GRID_FILE_SUFFIX}.parquet", index=False)

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


def test_area_metadata_places_the_id_raster(client: TestClient) -> None:
    """The metadata gives the ID raster's top-left corner, pixel size and shape."""
    response: Response = client.get(url="/areas/test")

    # Three 100 m segments in one row whose centres lie at y 50 span y 0 to 100
    assert response.json() == {
        "west": 0.0,
        "north": 100.0,
        "pixel_size_m": 100,
        "width": 3,
        "height": 1,
        "segment_count": 3,
    }


def test_segment_ids_png_holds_each_segment_id_at_its_pixel(client: TestClient) -> None:
    """The ID raster arrives as a PNG whose pixels decode to the segment IDs."""
    response: Response = client.get(url="/areas/test/segment_ids.png")

    assert response.headers["content-type"] == "image/png"
    rgb: np.ndarray = np.asarray(
        a=Image.open(fp=BytesIO(initial_bytes=response.content))
    )
    # IDs below 256 have only a low byte, so R and G are 0 and B is the ID
    assert rgb[..., 2].tolist() == [[0, 1, 2]]
    assert not rgb[..., :2].any()


def test_pixel_counts_returns_four_bytes_per_segment(client: TestClient) -> None:
    """Pixel counts arrive as uint32 in segment_id order."""
    response: Response = client.get(url="/areas/test/pixel_counts")

    assert response.content == np.array(object=[100, 100, 100], dtype="<u4").tobytes()


@pytest.mark.parametrize(
    argnames="url",
    argvalues=[
        "/areas/nowhere",
        "/areas/nowhere/segment_ids.png",
        "/areas/nowhere/pixel_counts",
        "/areas/nowhere/similar?segment=0",
        "/areas/nowhere/cloud?west=0&south=0&east=1&north=1",
    ],
)
def test_unknown_area_returns_404_on_every_area_endpoint(
    client: TestClient, url: str
) -> None:
    """An area without a vectors table is not found, whatever is asked about it."""
    response: Response = client.get(url=url)

    assert response.status_code == 404


@pytest.mark.parametrize(
    argnames=("segment", "status_code"),
    argvalues=[
        # One past the last segment
        (3, 404),
        # Negative IDs are rejected before the endpoint runs
        (-1, 422),
    ],
)
def test_similar_rejects_segments_outside_the_area(
    client: TestClient, segment: int, status_code: int
) -> None:
    """Searches for a segment that doesn't exist return an error status."""
    response: Response = client.get(
        url="/areas/test/similar", params={"segment": segment}
    )

    assert response.status_code == status_code


def test_cloud_returns_sorted_ids_then_positions_of_the_segments_in_the_bbox(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cloud holds 4 bytes per segment ID, then 3 position bytes per segment."""
    # The test area has only three segments, so allow a cloud of two
    monkeypatch.setattr(target=api, name="CLOUD_MIN_SEGMENTS", value=2)

    # The bbox holds the centres of segments 1 and 2 (x 150 and 250) but not of 0
    response: Response = client.get(
        url="/areas/test/cloud",
        params={"west": 100, "south": 0, "east": 300, "north": 100},
    )

    assert response.content[:8] == np.array(object=[1, 2], dtype="<u4").tobytes()
    assert len(response.content) == 2 * 4 + 2 * 3
    assert len(response.headers["x-explained-variance"].split(sep=",")) == 3


def test_cloud_refuses_a_selection_too_small_for_a_pca(client: TestClient) -> None:
    """A bbox with fewer segments than the minimum gets an error instead of a cloud."""
    response: Response = client.get(
        url="/areas/test/cloud",
        params={"west": 0, "south": 0, "east": 300, "north": 100},
    )

    assert response.status_code == 422
