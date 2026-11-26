"""Tests for loading boundaries and choosing their UTM zone."""

import json
from pathlib import Path
from typing import Any

import pytest
from rasterio.warp import transform_geom
from shapely import MultiPolygon, Point, Polygon, box
from shapely.geometry import mapping, shape
from shapely.geometry.base import BaseGeometry

from earth_similarity_explorer.boundary import (
    define_tiles,
    define_utm_epsg,
    load_boundary,
)

# Non-square shapes near Schlehdorf (a circle and two circles as one MultiPolygon)
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

    assert boundary.equals(other=geometry)


@pytest.mark.parametrize(
    argnames=("geometries", "message"),
    argvalues=[
        ([CIRCLE, CIRCLE], "exactly one feature"),
        ([Point(11.31, 47.65)], "Polygon or MultiPolygon"),
        # A bow tie whose ring crosses itself
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
        # Entirely in zone 32 (6° E to 12° E)
        (box(xmin=11.27, ymin=52, xmax=11.35, ymax=53), "EPSG:32632"),
        # Crosses 12° E, centroid at 12.5° E
        (box(xmin=11.0, ymin=52, xmax=14.0, ymax=53), "EPSG:32633"),
        # Crosses 12° E, centroid at 11.25° E
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


def test_tiles_overlap_the_boundary_and_run_north_to_south() -> None:
    """Tiles on a grid anchored at (0, 0) are listed if they overlap the boundary."""
    # Two 2 × 2 km squares in UTM zone 32N, each on a corner of four 20 km tiles,
    # given in longitude/latitude like a boundary file
    squares: MultiPolygon = MultiPolygon(
        polygons=[
            box(xmin=659_000, ymin=5_279_000, xmax=661_000, ymax=5_281_000),
            box(xmin=719_000, ymin=5_279_000, xmax=721_000, ymax=5_281_000),
        ]
    )
    boundary: Any = shape(
        transform_geom(src_crs="EPSG:32632", dst_crs="EPSG:4326", geom=mapping(squares))
    )

    tiles: list[tuple[int, int]] = define_tiles(
        boundary=boundary, crs="EPSG:32632", tile_size_m=20_000
    )

    # The tiles with their west edge at 680 000 m lie between the squares, so the
    # bounding box holds 10 tiles but only 8 are listed
    assert tiles == [
        (640_000, 5_300_000),
        (660_000, 5_300_000),
        (700_000, 5_300_000),
        (720_000, 5_300_000),
        (640_000, 5_280_000),
        (660_000, 5_280_000),
        (700_000, 5_280_000),
        (720_000, 5_280_000),
    ]
