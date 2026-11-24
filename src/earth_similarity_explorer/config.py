"""Study area configuration and validation."""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from earth_similarity_explorer.settings import (
    AREA_NAME_MAX_LENGTH,
    FIRST_EMBEDDING_YEAR,
    LAST_EMBEDDING_YEAR,
)


class AreaConfig(BaseModel):
    """The definition of a study area with id, name, target embedding year and boundary.

    The boundary path is relative to the repository root and not checked here.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    # Lowercase letters and digits joined by single hyphens, since the id becomes
    # part of folder and storage paths
    id: str = Field(pattern=r"^[a-z0-9]+(-[a-z0-9]+)*$")
    name: str = Field(min_length=1, max_length=AREA_NAME_MAX_LENGTH)
    year: int = Field(strict=True, ge=FIRST_EMBEDDING_YEAR, le=LAST_EMBEDDING_YEAR)
    boundary: Path


def load_area_config(config_yaml_path: Path) -> AreaConfig:
    """Read an area config from a YAML file and validate it.

    The file must be named after the area's id, e.g. ``areas/schlehdorf.yaml`` for
    the id ``schlehdorf``.

    Raises:
        FileNotFoundError: the file does not exist.
        pydantic.ValidationError: a field is missing, unknown or invalid.
        ValueError: the file name does not match the id.
    """
    text: str = config_yaml_path.read_text(encoding="utf-8")
    data: Any = yaml.safe_load(stream=text)
    config: AreaConfig = AreaConfig.model_validate(obj=data)

    # A copied file with an unchanged id would overwrite another area's outputs
    if config.id != config_yaml_path.stem:
        raise ValueError(
            f"{config_yaml_path}: id {config.id!r} does not match the file name "
            f"{config_yaml_path.stem!r}"
        )

    return config
