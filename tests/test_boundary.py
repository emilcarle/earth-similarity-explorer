import json
from pathlib import Path

import pytest
from shapely import MultiPolygon, Point, Polygon, box
from shapely.geometry import mapping
from shapely.geometry.base import BaseGeometry

from earth_similarity_explorer.boundary import define_utm_zone, load_boundary
from earth_similarity_explorer.config import AreaConfig, load_area_config

REPOSITORY_ROOT: Path = Path(__file__).parents[1]

# Non-square shapes near Schlehdorf: a circle, and two circles as one MultiPolygon
CIRCLE: Polygon = Point(11.31, 47.65).buffer(distance=0.03)
TWO_CIRCLES: MultiPolygon = MultiPolygon(
    polygons=[CIRCLE, Point(11.45, 47.65).buffer(distance=0.03)]
)


# HELPER FUNCTIONS
def write_geojson(directory: Path, geometries: list[BaseGeometry]) -> Path:
    """Write the geometries as a GeoJSON FeatureCollection, one feature each."""
    features: list[dict[str, object]] = [
        {"type": "Feature", "properties": {}, "geometry": mapping(geometry)}
        for geometry in geometries
    ]
    path: Path = directory / "boundary.geojson"
    path.write_text(
        data=json.dumps(obj={"type": "FeatureCollection", "features": features}),
        encoding="utf-8",
    )
    return path


# TESTS
@pytest.mark.parametrize(
    argnames="geometry",
    argvalues=[CIRCLE, TWO_CIRCLES],
    ids=["polygon", "multipolygon"],
)
def test_valid_boundary_loads(tmp_path: Path, geometry: BaseGeometry) -> None:
    """A valid (Multi)Polygon in longitude/latitude loads unchanged."""
    boundary: Polygon | MultiPolygon = load_boundary(
        path=write_geojson(directory=tmp_path, geometries=[geometry])
    )

    assert boundary.equals(geometry)


@pytest.mark.parametrize(
    argnames=("geometries", "message"),
    argvalues=[
        ([CIRCLE, CIRCLE], "exactly one feature"),
        ([Point(11.31, 47.65)], "Polygon or MultiPolygon"),
        # A bow tie: the ring crosses itself
        (
            [Polygon(shell=[(11.3, 47.6), (11.4, 47.7), (11.4, 47.6), (11.3, 47.7)])],
            "not valid",
        ),
        # UTM metres instead of degrees, e.g. a file in EPSG:32632
        (
            [box(xmin=660000, ymin=5280000, xmax=665000, ymax=5285000)],
            "longitude/latitude",
        ),
    ],
    ids=["two-features", "point", "self-intersecting", "utm-coordinates"],
)
def test_invalid_boundary_fails(
    tmp_path: Path, geometries: list[BaseGeometry], message: str
) -> None:
    """An unusable boundary file fails with a message naming the problem."""
    with pytest.raises(expected_exception=ValueError, match=message):
        load_boundary(path=write_geojson(directory=tmp_path, geometries=geometries))


@pytest.mark.parametrize(
    argnames=("west", "east", "zone"),
    argvalues=[
        (11.27, 11.35, 32),  # entirely in zone 32 (6° E to 12° E)
        (11.0, 14.0, 33),  # crosses 12° E, centroid at 12.5° E
        (10.0, 12.5, 32),  # crosses 12° E, centroid at 11.25° E
    ],
    ids=["one-zone", "mostly-east", "mostly-west"],
)
def test_define_utm_zone_contains_centroid(west: float, east: float, zone: int) -> None:
    """The UTM zone is the one containing the area's centroid."""
    boundary: Polygon = box(xmin=west, ymin=52, xmax=east, ymax=53)

    assert define_utm_zone(boundary=boundary) == zone


@pytest.mark.parametrize(
    argnames="config_path",
    argvalues=sorted((REPOSITORY_ROOT / "areas").glob(pattern="*.yaml")),
    ids=lambda p: p.stem,
)
def test_repository_boundaries_are_valid(config_path: Path) -> None:
    """Every area config in /areas points to a valid boundary file."""
    config: AreaConfig = load_area_config(config_yaml_path=config_path)

    load_boundary(path=REPOSITORY_ROOT / config.boundary)
