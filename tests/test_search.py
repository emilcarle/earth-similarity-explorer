"""Tests for the similarity search."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from earth_similarity_explorer.search import (
    encode_scores,
    load_vectors,
    score_similarity,
)
from earth_similarity_explorer.settings import EMBEDDING_BANDS


# TESTS
@pytest.mark.parametrize(
    argnames=("query_id", "expected"),
    argvalues=[
        # Segment 0 against itself, a right angle, the opposite and 45°
        (0, [1.0, 0.0, -1.0, 0.7071]),
        # Another query whose scores stay in segment_id order
        (1, [0.0, 1.0, 0.0, 0.7071]),
    ],
)
def test_scores_are_cosines_in_segment_id_order(
    query_id: int, expected: list[float]
) -> None:
    """Each score is the cosine between a segment and the query, at the segment's position."""
    # Four segments using only A00 and A01 that point east, north, west and north-east
    vectors: np.ndarray = np.zeros(shape=(4, len(EMBEDDING_BANDS)), dtype=np.int8)
    vectors[:, :2] = [[127, 0], [0, 127], [-127, 0], [90, 90]]

    scores: np.ndarray = score_similarity(vectors=vectors, query_id=query_id)

    # The int8 rounding of [90, 90] makes the 45° score 0.709 instead of 0.707
    assert scores.tolist() == pytest.approx(expected=expected, abs=0.002)


def test_load_vectors_returns_the_int8_matrix_in_segment_id_order(
    tmp_path: Path,
) -> None:
    """Only the 64 vector columns come back, as int8, with row number = segment_id."""
    # Three segments whose A00 values 10, 20, 30 show the row order
    table: pd.DataFrame = pd.DataFrame(
        data={
            "segment_id": np.arange(3, dtype=np.int32),
            "x": [50.0, 150.0, 250.0],
            "y": [50.0, 50.0, 50.0],
            "pixel_count": np.array(object=[100, 100, 37], dtype=np.int32),
            "homogeneity": np.array(object=[0.99, 0.95, 0.9], dtype=np.float32),
            **{band: np.zeros(shape=3, dtype=np.int8) for band in EMBEDDING_BANDS},
        }
    )
    table["A00"] = np.array(object=[10, 20, 30], dtype=np.int8)
    path: Path = tmp_path / "vectors.parquet"
    table.to_parquet(path=path, index=False)

    vectors: np.ndarray = load_vectors(path=path)

    assert vectors.shape == (3, 64)
    assert vectors.dtype == np.int8
    assert vectors[:, 0].tolist() == [10, 20, 30]


def test_encode_scores_maps_minus_one_to_one_onto_bytes() -> None:
    """Scores become bytes 0 to 255, and scores just outside −1 to 1 stay at the ends."""
    # The ends, the middle, an example (0.93) and a rounded-up self-score
    scores: np.ndarray = np.array(object=[-1.0, 0.0, 0.93, 1.0, 1.016])

    assert list(encode_scores(scores=scores)) == [0, 128, 246, 255, 255]
