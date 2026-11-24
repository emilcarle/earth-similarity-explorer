"""Tests for the data the web page needs besides the scores."""

from io import BytesIO

import numpy as np
from PIL import Image

from earth_similarity_explorer.serving import build_id_raster, encode_id_png
from earth_similarity_explorer.settings import NO_SEGMENT_ID


# TESTS
def test_id_raster_puts_each_segment_id_into_its_pixel() -> None:
    """Each segment fills the pixel at its centre, and pixels without one hold NO_SEGMENT_ID."""
    # Three 100 m segments in an L shape, in segment_id order (north to south, west to east)
    x: np.ndarray = np.array(object=[50.0, 150.0, 50.0])
    y: np.ndarray = np.array(object=[150.0, 150.0, 50.0])

    ids, corner = build_id_raster(x=x, y=y, pixel_size_m=100)

    assert ids.tolist() == [[0, 1], [2, NO_SEGMENT_ID]]
    # The top-left corner of the raster is half a pixel beyond the outermost centres
    assert corner == (0.0, 200.0)


def test_id_png_keeps_every_id_exact_as_big_endian_rgb() -> None:
    """Decoding the PNG gives back the IDs, with the high byte in R and the low byte in B."""
    # 70,000 is 0x011170, so its bytes are R 1, G 17, B 112
    ids: np.ndarray = np.array(
        object=[[0, 70_000], [NO_SEGMENT_ID - 1, NO_SEGMENT_ID]], dtype=np.uint32
    )

    image: Image.Image = Image.open(fp=BytesIO(encode_id_png(ids=ids)))
    rgb: np.ndarray = np.asarray(image).astype(dtype=np.uint32)

    # RGB without an alpha channel
    assert image.mode == "RGB"
    assert rgb[0, 1].tolist() == [1, 17, 112]
    decoded: np.ndarray = (rgb[..., 0] << 16) | (rgb[..., 1] << 8) | rgb[..., 2]
    assert decoded.tolist() == ids.tolist()
