"""Project settings: fixed values, and values read from the environment or .env.

Tuned parameters (e.g. segmentation) do not belong here but in params.yaml, so that
DVC can track them.
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# FIXED VALUES: identical on every machine, changed only by a code change.
# Plain constants, not Settings fields: readable without .env (e.g. in CI), and no
# environment variable can override them.

# Project folders, relative to the project root, where DVC runs every stage
AREAS_DIR: Path = Path("areas")
DATA_DIR: Path = Path("data")
GRID_CLUSTERS_DIR: Path = DATA_DIR / "grid_clusters"

# The AlphaEarth annual embeddings in Earth Engine, and the years they cover.
# LAST_EMBEDDING_YEAR should be raised by hand when Google publishes a new year:
# https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_SATELLITE_EMBEDDING_V1_ANNUAL
EMBEDDING_COLLECTION_ID: str = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"
FIRST_EMBEDDING_YEAR: int = 2017
LAST_EMBEDDING_YEAR: int = 2024
# Pixel size of the alphaearth embeddings in m
EMBEDDING_PIXEL_SIZE_M: int = 10
# Embedding values can be stored as int8 with a minor loss in precision: round(value × scale)
EMBEDDING_INT8_SCALE: int = 127
# Google Drive folder that Earth Engine exports to
DRIVE_EXPORT_FOLDER: str = "earth-similarity-explorer"
# The Drive REST API (v3): one URL to list files, one per file to download or delete
DRIVE_FILES_URL: str = "https://www.googleapis.com/drive/v3/files"

# Frontend Specifications
# Max length for the area name (rendered in the application)
AREA_NAME_MAX_LENGTH: int = 100


# ENVIRONMENT: secrets and machine-specific values
class Settings(BaseSettings):
    """The keys listed in .env.example. Field names match them case-insensitively."""

    # Ignoring extra keys during development
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", frozen=True)

    ee_project: str
