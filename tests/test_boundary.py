import json
from pathlib import Path

import pytest
from shapely import MultiPolygon, Point, Polygon, box
from shapely.geometry import mapping
from shapely.geometry.base import BaseGeometry

from earth_similarity_explorer.boundary import define_utm_epsg, load_boundary

# Non-square shapes near Schlehdorf: a circle, and two circles as one MultiPolygon
CIRCLE: Polygon = Point(11.31, 47.65).buffer(distance=0.03)
TWO_CIRCLES: MultiPolygon = MultiPolygon(
    polygons=[CIRCLE, Point(11.45, 47.65).buffer(distance=0.03)]
)


# HELPER FUNCTIONS
def write_geojson(directory: Path, geometries: list[BaseGeometry]) -> Path:
    """Write the geometries as a GeoJSON FeatureCollection, one feature each."""
    features: list[dict[str, object]] = [
        {"type": "Feature", "properties": {}, "geometry": mapping(ob=geometry)}
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
    argnames=("boundary", "epsg"),
    argvalues=[
        # entirely in zone 32 (6° E to 12° E)
        (box(xmin=11.27, ymin=52, xmax=11.35, ymax=53), "EPSG:32632"),
        # crosses 12° E, centroid at 12.5° E
        (box(xmin=11.0, ymin=52, xmax=14.0, ymax=53), "EPSG:32633"),
        # crosses 12° E, centroid at 11.25° E
        (box(xmin=10.0, ymin=52, xmax=12.5, ymax=53), "EPSG:32632"),
        # Sydney, southern hemisphere
        (box(xmin=151.0, ymin=-34, xmax=151.3, ymax=-33.7), "EPSG:32756"),
    ],
    ids=["one-zone", "mostly-east", "mostly-west", "southern-hemisphere"],
)
def test_define_utm_epsg_contains_centroid(
    boundary: Polygon | MultiPolygon, epsg: str
) -> None:
    """The EPSG code is the UTM zone and hemisphere containing the area's centroid."""
    assert define_utm_epsg(boundary=boundary) == epsg
