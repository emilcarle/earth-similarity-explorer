"""Data backend delivery for the application, including segment IDs."""

from io import BytesIO

import numpy as np
from PIL import Image

from earth_similarity_explorer.settings import NO_SEGMENT_ID


def build_id_raster(
    x: np.ndarray, y: np.ndarray, pixel_size_m: float
) -> tuple[np.ndarray, tuple[float, float]]:
    """
    Return a raster holding each segment's ID and its top-left corner in UTM metres.
    x and y are the segment centres in segment_id order. Pixels without a segment hold NO_SEGMENT_ID.
    """
    # A larger ID would collide with NO_SEGMENT_ID or not fit into 24 bits
    if len(x) >= NO_SEGMENT_ID:
        raise ValueError(f"{len(x)} segments do not fit into 24-bit IDs")

    # Column and row of each centre, counted from the western and northern edge
    west: float = float(x.min()) - pixel_size_m / 2
    north: float = float(y.max()) + pixel_size_m / 2
    cols: np.ndarray = np.floor((x - west) / pixel_size_m).astype(dtype=np.int64)
    rows: np.ndarray = np.floor((north - y) / pixel_size_m).astype(dtype=np.int64)

    # Initially fill with NO_SEGMENT_ID everywhere
    ids: np.ndarray = np.full(
        shape=(rows.max() + 1, cols.max() + 1),
        fill_value=NO_SEGMENT_ID,
        dtype=np.uint32,
    )
    # Write each segment_id into the pixel at its row and column
    ids[rows, cols] = np.arange(len(x), dtype=np.uint32)
    return ids, (west, north)


def encode_id_png(ids: np.ndarray) -> bytes:
    """Encode a raster of 24-bit IDs as an RGB PNG."""
    # Split each ID into its high, middle and low byte, as the last axis (height × width × 3)
    rgb: np.ndarray = np.stack(
        arrays=[(ids >> 16) & 255, (ids >> 8) & 255, ids & 255], axis=-1
    ).astype(dtype=np.uint8)

    # Write an RGB PNG into memory. Pillow adds no alpha channel or colour profile, and
    # optimize finds the smallest lossless compression.
    buffer: BytesIO = BytesIO()
    Image.fromarray(obj=rgb).save(fp=buffer, format="PNG", optimize=True)
    return buffer.getvalue()
