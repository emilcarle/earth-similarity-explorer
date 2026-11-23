from pathlib import Path
from typing import Any

import ee
import pytest
from pydantic import ValidationError
from shapely import MultiPolygon, Polygon
from shapely.geometry import mapping

from earth_similarity_explorer.boundary import define_utm_epsg, load_boundary
from earth_similarity_explorer.config import AreaConfig, load_area_config
from earth_similarity_explorer.earth_engine import initialize_earth_engine
from earth_similarity_explorer.settings import EMBEDDING_COLLECTION_ID, Settings

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
    facts: Any | None = ee.Dictionary(
        arg={
            "image_count": images.size(),
            "bands": first_image.bandNames(),
            "crs": first_image.projection().crs(),
        }
    ).getInfo()

    assert facts is not None
    assert facts["image_count"] >= 1
    assert facts["bands"] == [f"A{band:02d}" for band in range(64)]
    assert facts["crs"] == define_utm_epsg(boundary=boundary)
