"""Connection to Google Earth Engine."""

import ee

from earth_similarity_explorer.settings import Settings


def initialize_earth_engine() -> None:
    """Connect to Earth Engine with the Cloud project from the settings."""
    ee.Initialize(project=Settings().ee_project)
