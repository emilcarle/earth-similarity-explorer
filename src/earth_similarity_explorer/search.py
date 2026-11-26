"""Similarity search over an area's vectors table."""

from pathlib import Path

import numpy as np
import pandas as pd

from earth_similarity_explorer.settings import EMBEDDING_BANDS, SCORE_BYTE_MAX


def load_vectors(path: Path) -> np.ndarray:
    """Read the float32 unit vectors (N × 64) of a vectors table in segment_id order."""
    # Read only the 64 vector columns because search needs nothing else
    vectors: pd.DataFrame = pd.read_parquet(path=path, columns=EMBEDDING_BANDS)
    # The rows are stored in segment_id order, so row number i is segment i
    return vectors.to_numpy()


def score_similarity(vectors: np.ndarray, query_id: int) -> np.ndarray:
    """Return the similarity of every segment to the query segment, in segment_id order.

    vectors holds unit vectors (N × 64), where the row number is the segment_id.
    The scores are cosine similarities, between −1 and 1.
    """
    # Calculate the dot product of every vector with the query vector (cosine for unit vectors)
    return vectors @ vectors[query_id]


def encode_scores(scores: np.ndarray) -> bytes:
    """Turn scores between −1 and 1 into one byte each, in the same order."""
    # Map −1…1 to 0…255 so 0.93 becomes round(1.93 / 2 × 255) = 246
    levels: np.ndarray = np.round(a=(scores + 1) / 2 * SCORE_BYTE_MAX)
    return levels.astype(dtype=np.uint8).tobytes()
