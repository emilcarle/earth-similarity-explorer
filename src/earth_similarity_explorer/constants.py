"""
Fixed facts, constants and rules shared across the project.
Does not include segmentation settings that belong in params.yaml so that DVC can track it.
"""

# Years covered by the AlphaEarth annual embeddings. Raise LAST_EMBEDDING_YEAR by hand
# when Google publishes a new year:
# https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_SATELLITE_EMBEDDING_V1_ANNUAL
FIRST_EMBEDDING_YEAR: int = 2017
LAST_EMBEDDING_YEAR: int = 2024

# Max length for the area name (rendered in the application)
AREA_NAME_MAX_LENGTH: int = 100
