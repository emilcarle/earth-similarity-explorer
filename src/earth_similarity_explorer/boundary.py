"""Loading and checking boundary geometries, choosing their UTM zone and export tiles."""

import math
from pathlib import Path

import shapely
from rasterio.warp import transform_geom
from shapely import GeometryCollection, MultiPolygon, Point, Polygon
from shapely.geometry import mapping, shape
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
    # An invalid geometry, e.g. a ring that crosses itself, fails with shapely's reason
    if not geometry.is_valid:
        raise ValueError(
            f"{path}: not valid, {shapely.is_valid_reason(geometry=geometry)}"
        )
    # Only areas can be study areas, not points or lines
    if isinstance(geometry, Polygon | MultiPolygon):
        return geometry
    raise ValueError(f"{path}: expected a Polygon or MultiPolygon")


def define_utm_epsg(boundary: Polygon | MultiPolygon) -> str:
    """Return the EPSG code of the UTM zone containing the boundary's centroid."""
    centroid: Point = boundary.centroid
    # Zones are 6° wide, numbered from 1 at 180° W. EPSG 326xx is north, 327xx south
    zone: int = int((centroid.x + 180) // 6) + 1
    return f"EPSG:{326 if centroid.y >= 0 else 327}{zone:02d}"


def define_tiles(
    boundary: Polygon | MultiPolygon, crs: str, tile_size_m: int
) -> list[tuple[int, int]]:
    """Defines tiles for Earth Engine exports based on the boundary geometry and the tile size

    Tiles are squares on a grid anchored at (0, 0), listed north to south, then west
    to east.
    """
    # Reproject the boundary from longitude/latitude (EPSG:4326) to the crs
    projected: BaseGeometry = shape(
        transform_geom(src_crs="EPSG:4326", dst_crs=crs, geom=mapping(ob=boundary))
    )
    # Snap the bounding box outward to the tile grid
    west, south, east, north = projected.bounds
    first_west: int = math.floor(west / tile_size_m) * tile_size_m
    first_north: int = math.ceil(north / tile_size_m) * tile_size_m
    last_east: int = math.ceil(east / tile_size_m) * tile_size_m
    last_south: int = math.floor(south / tile_size_m) * tile_size_m
    # Keep the tiles that share area with the boundary, not only an edge
    tiles: list[tuple[int, int]] = []
    for tile_north in range(first_north, last_south, -tile_size_m):
        for tile_west in range(first_west, last_east, tile_size_m):
            tile: Polygon = shapely.box(
                xmin=tile_west,
                ymin=tile_north - tile_size_m,
                xmax=tile_west + tile_size_m,
                ymax=tile_north,
            )
            if tile.intersects(projected) and not tile.touches(projected):
                tiles.append((tile_west, tile_north))
    return tiles
