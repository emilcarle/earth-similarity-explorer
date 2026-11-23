"""Pipeline parameters from params.yaml, and their validation."""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from earth_similarity_explorer.settings import EMBEDDING_PIXEL_SIZE_M


class GridParams(BaseModel):
    """The grid baseline: square cells of mean embeddings."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    # Whole pixels only, so every cell edge lies on the embedding pixel grid
    cell_size_m: int = Field(strict=True, gt=0, multiple_of=EMBEDDING_PIXEL_SIZE_M)


class Params(BaseModel):
    """All tuned parameters, one section per pipeline stage."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    areas: list[str] = Field(min_length=1)
    grid: GridParams


def load_params(params_yaml_path: Path = Path("params.yaml")) -> Params:
    """Read and validate params.yaml (by default the one in the project root)."""
    data: Any = yaml.safe_load(stream=params_yaml_path.read_text(encoding="utf-8"))
    return Params.model_validate(obj=data)
