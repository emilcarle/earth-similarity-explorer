"""Live checks against Earth Engine, skipped without credentials."""

from pathlib import Path
from typing import Any

import ee
import numpy as np
import pytest
from pydantic import ValidationError
from shapely import MultiPolygon, Polygon
from shapely.geometry import mapping

from earth_similarity_explorer.boundary import define_utm_epsg, load_boundary
from earth_similarity_explorer.config import AreaConfig, load_area_config
from earth_similarity_explorer.earth_engine import (
    decode_int8,
    encode_int8,
    initialize_earth_engine,
)
from earth_similarity_explorer.settings import (
    EMBEDDING_BANDS,
    EMBEDDING_COLLECTION_ID,
    EXPORT_INT8_NODATA,
    Settings,
)

REPOSITORY_ROOT: Path = Path(__file__).parents[1]

# These tests call Earth Engine, so they need credentials and EE_PROJECT in .env.
# Without them (e.g. in CI) the whole module is skipped
try:
    Settings()
except ValidationError:
    pytest.skip(reason="EE_PROJECT is not set", allow_module_level=True)


# TESTS
def test_embeddings_cover_fixture_area_in_its_utm_zone() -> None:
    """AlphaEarth has 64-band images over the fixture area."""
    initialize_earth_engine()
    config: AreaConfig = load_area_config(
        config_yaml_path=REPOSITORY_ROOT / "areas" / "schlehdorf.yaml"
    )
    boundary: Polygon | MultiPolygon = load_boundary(
        path=REPOSITORY_ROOT / config.boundary
    )

    images: ee.ImageCollection = (
        ee.ImageCollection(args=EMBEDDING_COLLECTION_ID)
        .filterBounds(geometry=ee.Geometry(geo_json=mapping(ob=boundary)))
        .filterDate(start=f"{config.year}-01-01", end=f"{config.year + 1}-01-01")
    )
    first_image: ee.Image = images.first()
    # One request to Earth Engine for all three facts
    facts: Any = ee.Dictionary(
        arg={
            "image_count": images.size(),
            "bands": first_image.bandNames(),
            "crs": first_image.projection().crs(),
        }
    ).getInfo()

    assert facts["image_count"] >= 1
    assert facts["bands"] == EMBEDDING_BANDS
    assert facts["crs"] == define_utm_epsg(boundary=boundary)


def test_int8_encoding_round_trips_every_value_and_marks_masked_pixels() -> None:
    """Every AlphaEarth value survives the int8 export exactly, and masked pixels become no-data."""
    initialize_earth_engine()
    # AlphaEarth's values, written out independently of the module's constants
    integers: np.ndarray = np.arange(-127, stop=128, dtype=np.int8)
    values: np.ndarray = (np.sign(integers) * (integers / 127.5) ** 2).astype(
        dtype=np.float32
    )

    # One band per value, then one fully masked band, sampled at one pixel
    image: ee.Image = ee.Image.constant(value=values.tolist()).addBands(
        srcImg=ee.Image.constant(value=0).updateMask(mask=0).rename("masked")
    )
    sample: ee.Dictionary = encode_int8(embeddings=image).reduceRegion(
        reducer=ee.Reducer.first(),
        geometry=ee.Geometry.Point(coords=[11.3, 47.6]),
        scale=10,
    )
    stored: np.ndarray = np.array(
        object=sample.values(keys=image.bandNames()).getInfo(), dtype=np.int8
    )

    assert stored.tolist() == [*integers.tolist(), EXPORT_INT8_NODATA]
    # NaN counts as equal to NaN here, so the masked band must decode to NaN
    np.testing.assert_array_equal(
        actual=decode_int8(stored=stored), desired=[*values, np.nan]
    )
