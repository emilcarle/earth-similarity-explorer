"""Study area configuration and validation."""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from earth_similarity_explorer.constants import (
    AREA_NAME_MAX_LENGTH,
    FIRST_EMBEDDING_YEAR,
    LAST_EMBEDDING_YEAR,
)


class AreaConfig(BaseModel):
    """
    The definition of a study area: id, name, target embedding year, boundary geometry.
    The paths are relative to the repository root and not checked by pydantic here.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    # Id should be lowercase characters joined by single hyphens.
    # The id becomes folder and storage paths.
    id: str = Field(pattern=r"^[a-z0-9]+(-[a-z0-9]+)*$")
    name: str = Field(min_length=1, max_length=AREA_NAME_MAX_LENGTH)
    year: int = Field(strict=True, ge=FIRST_EMBEDDING_YEAR, le=LAST_EMBEDDING_YEAR)
    boundary: Path


def load_area_config(config_yaml_path: Path) -> AreaConfig:
    """Read an area config from a YAML file and validate it.

    The file must be named after the area's id, e.g. ``areas/munich-garmisch.yaml``
    for the id ``munich-garmisch``.

    Raises:
        FileNotFoundError: the file does not exist.
        pydantic.ValidationError: a field is missing, unknown or invalid.
        ValueError: the file name does not match the id.
    """
    text: str = config_yaml_path.read_text(encoding="utf-8")
    data: Any = yaml.safe_load(stream=text)
    config: AreaConfig = AreaConfig.model_validate(obj=data)

    # Ensure that copied files with old id's are not accidently overwriting other study areas.
    if config.id != config_yaml_path.stem:
        raise ValueError(
            f"{config_yaml_path}: id {config.id!r} does not match the file name "
            f"{config_yaml_path.stem!r}"
        )

    return config
