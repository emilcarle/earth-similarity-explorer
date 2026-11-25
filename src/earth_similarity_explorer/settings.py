"""Project settings with fixed values and values read from the environment or .env.

Tuned parameters (e.g. segmentation) do not belong here but in params.yaml, so that
DVC can track them.
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# FIXED VALUES are identical on every machine and change only by a code change.
# As plain constants instead of Settings fields they are readable without .env
# (e.g. in CI) and no environment variable can override them.

# Project files and folders, relative to the project root, where DVC runs every stage
PARAMS_PATH: Path = Path("params.yaml")
AREAS_DIR: Path = Path("areas")
DATA_DIR: Path = Path("data")
GRID_SEGMENTATION_DIR: Path = DATA_DIR / "grid_segmentation"
VECTORS_DIR: Path = DATA_DIR / "vectors"
# Output files of the grid segmentation end in this suffix, e.g. schlehdorf_grid.tif.
# The output paths in dvc.yaml repeat it
GRID_FILE_SUFFIX: str = "_grid"

# The AlphaEarth annual embeddings in Earth Engine, and the years they cover.
# Raise LAST_EMBEDDING_YEAR by hand when Google publishes a new year on this page
# https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_SATELLITE_EMBEDDING_V1_ANNUAL
EMBEDDING_COLLECTION_ID: str = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"
FIRST_EMBEDDING_YEAR: int = 2017
LAST_EMBEDDING_YEAR: int = 2024
# Pixel size of the AlphaEarth embeddings in metres
EMBEDDING_PIXEL_SIZE_M: int = 10
# Names of the 64 embedding bands (A00 to A63), which are also the vector columns
# of the tables
EMBEDDING_BANDS: list[str] = [f"A{band:02d}" for band in range(64)]
# Name of the band that holds each segment's pixel count in a segmentation GeoTIFF
PIXEL_COUNT_BAND: str = "pixel_count"
# Embedding values can be stored as int8 with a minor loss in precision as round(value × scale)
EMBEDDING_INT8_SCALE: int = 127
# Google Drive folder that Earth Engine exports to
DRIVE_EXPORT_FOLDER: str = "earth-similarity-explorer"
# The Drive REST API (v3) lists files at this URL and serves each file at URL/<id>
DRIVE_FILES_URL: str = "https://www.googleapis.com/drive/v3/files"
# The API sends each score as one byte, with −1 as 0 and 1 as this value
SCORE_BYTE_MAX: int = 255
# The API sends each 3D cloud position as one byte per axis. The centre of the cloud
# becomes 128 and the clipping range becomes 128 ± 127.
POSITION_BYTE_CENTRE: int = 128
POSITION_BYTE_HALF_RANGE: int = 127
# Positions beyond this percentile of all absolute coordinates are clipped, so a few
# outliers don't squeeze the rest of the cloud together
POSITION_CLIP_PERCENTILE: float = 99.9
# The PCA of a cloud is fitted to at most this many randomly drawn segments of the
# selection. The fixed seed draws the same ones for the same selection.
CLOUD_FIT_SAMPLE_SIZE: int = 50_000
CLOUD_FIT_SEED: int = 0
# Below this many segments a PCA is degenerate or unstable (a 100 m pan turns its axes),
# so the API refuses to build a cloud
CLOUD_MIN_SEGMENTS: int = 100
# Segment IDs are packed into 24 bits (R, G, B), and the largest value marks "no segment"
NO_SEGMENT_ID: int = 2**24 - 1

# FRONTEND
# Longest area name the web page shows
AREA_NAME_MAX_LENGTH: int = 100


# ENVIRONMENT VALUES (secrets and machine-specific values)
class Settings(BaseSettings):
    """The keys listed in .env.example. Field names match them case-insensitively."""

    # Ignoring extra keys during development
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", frozen=True)

    ee_project: str
