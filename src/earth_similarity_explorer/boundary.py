"""Loading and checking the boundary geometries and deciding on the projected UTM."""

from pathlib import Path

import shapely
from shapely import GeometryCollection, MultiPolygon, Point, Polygon
from shapely.geometry.base import BaseGeometry


def load_boundary(path: Path) -> Polygon | MultiPolygon:
    """Read a boundary GeoJSON holding one valid (Multi)Polygon in longitude/latitude."""
    geometry: BaseGeometry = shapely.from_geojson(
        geometry=path.read_text(encoding="utf-8")
    )

    # A FeatureCollection becomes a GeometryCollection with one part per feature
    if isinstance(geometry, GeometryCollection):
        if len(geometry.geoms) != 1:
            raise ValueError(f"{path}: expected exactly one feature")
        geometry = geometry.geoms[0]

    # Coordinates beyond ±180°/±90° are not degrees, e.g. UTM metres
    if not shapely.box(xmin=-180, ymin=-90, xmax=180, ymax=90).covers(geometry):
        raise ValueError(f"{path}: coordinates are not longitude/latitude")
    if not geometry.is_valid:
        raise ValueError(f"{path}: not valid, {shapely.is_valid_reason(geometry)}")
    if isinstance(geometry, Polygon | MultiPolygon):
        return geometry
    raise ValueError(f"{path}: expected a Polygon or MultiPolygon")


def define_utm_epsg(boundary: Polygon | MultiPolygon) -> str:
    """Return the EPSG code of the UTM zone containing the boundary's centroid."""
    centroid: Point = boundary.centroid
    # Zones are 6° wide, numbered from 1 at 180° W. EPSG 326xx is north, 327xx south
    zone: int = int((centroid.x + 180) // 6) + 1
    return f"EPSG:{326 if centroid.y >= 0 else 327}{zone:02d}"
