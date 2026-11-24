"""Exact similarity search over an area's vectors table."""

from pathlib import Path

import numpy as np
import pandas as pd

from earth_similarity_explorer.settings import (
    EMBEDDING_BANDS,
    EMBEDDING_INT8_SCALE,
    SCORE_BYTE_MAX,
)


def load_vectors(path: Path) -> np.ndarray:
    """Read the int8 vectors (N × 64) of a vectors table in segment_id order."""
    # Read only the 64 vector columns because search needs nothing else
    vectors: pd.DataFrame = pd.read_parquet(path=path, columns=EMBEDDING_BANDS)
    # The rows are stored in segment_id order, so row number i is segment i
    return vectors.to_numpy()


def score_similarity(vectors: np.ndarray, query_id: int) -> np.ndarray:
    """Return the similarity of every segment to the query segment, in segment_id order.

    vectors holds int8 vectors (N × 64), where the row number is the segment_id.
    The scores are cosine similarities, between (approx.) −1 and 1.
    """
    # Convert to float32 since int8 products overflow (57 × 57 > 127)
    matrix: np.ndarray = vectors.astype(dtype=np.float32)
    # Calculate dot products
    dot_products: np.ndarray = matrix @ matrix[query_id]
    # Stored vectors are unit vectors × 127, so dividing by 127² gives the cosine
    return dot_products / EMBEDDING_INT8_SCALE**2


def encode_scores(scores: np.ndarray) -> bytes:
    """Turn scores between −1 and 1 into one byte each, in the same order."""
    # Map −1…1 to 0…255 so 0.93 becomes round(1.93 / 2 × 255) = 246
    levels: np.ndarray = np.round(a=(scores + 1) / 2 * SCORE_BYTE_MAX)
    # Self-scores can reach 1.016 because int8 rounding leaves stored lengths of 126–128.
    # Clipping keeps them at 255 instead of wrapping around to a byte near 0.
    clipped: np.ndarray = np.clip(a=levels, a_min=0, a_max=SCORE_BYTE_MAX)
    return clipped.astype(dtype=np.uint8).tobytes()
