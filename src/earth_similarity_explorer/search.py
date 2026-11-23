"""Similarity search for vector tables based on input vectors."""

from pathlib import Path

import numpy as np
import pandas as pd

from earth_similarity_explorer.settings import EMBEDDING_BANDS, EMBEDDING_INT8_SCALE


def load_vectors(path: Path) -> np.ndarray:
    """Read the int8 vectors (N × 64) of a vectors table in unit_id order."""
    # Read only the 64 vector columns because search needs nothing else
    vectors: pd.DataFrame = pd.read_parquet(path=path, columns=EMBEDDING_BANDS)
    # The rows are stored in unit_id order, so row number i is unit i
    return vectors.to_numpy()


def score_similarity(vectors: np.ndarray, query_id: int) -> np.ndarray:
    """Return the similarity of every unit to the query unit, in unit_id order.
    Vectors holds int8 vectors (N × 64), where the row number is the unit_id.
    The scores are cosine similarities, between (approx.) −1 and 1.
    """
    # Convert to float32 since int8 products overflow (57 × 57 > 127)
    matrix: np.ndarray = vectors.astype(dtype=np.float32)
    # Calculate dot products
    dot_products: np.ndarray = matrix @ matrix[query_id]
    # Stored vectors are unit vectors × 127, so dividing by 127² gives the cosine
    return dot_products / EMBEDDING_INT8_SCALE**2
