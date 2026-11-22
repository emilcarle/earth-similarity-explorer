"""Project settings: fixed values, and values read from the environment or .env.

Tuned parameters (e.g. segmentation) do not belong here but in params.yaml, so that
DVC can track them.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict

# FIXED VALUES: identical on every machine, changed only by a code change.
# Reading without .env (relevant for CI)

# The AlphaEarth annual embeddings in Earth Engine, and the years they cover. Raise
# LAST_EMBEDDING_YEAR by hand when Google publishes a new year:
# https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_SATELLITE_EMBEDDING_V1_ANNUAL
EMBEDDING_COLLECTION_ID: str = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"
FIRST_EMBEDDING_YEAR: int = 2017
LAST_EMBEDDING_YEAR: int = 2024

# Max length for the area name (rendered in the application)
AREA_NAME_MAX_LENGTH: int = 100


# ENVIRONMENT: secrets and machine-specific values
class Settings(BaseSettings):
    """The keys listed in .env.example; field names match them case-insensitively."""

    # Ignoring extra keys during development
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", frozen=True)

    ee_project: str
