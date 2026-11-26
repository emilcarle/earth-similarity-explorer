"""Pipeline parameters from params.yaml, and their validation."""

from pathlib import Path
from typing import Any, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from earth_similarity_explorer.settings import EMBEDDING_PIXEL_SIZE_M, PARAMS_PATH


class ExportParams(BaseModel):
    """The export of the embeddings from Earth Engine in square tiles."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    # Whole pixels only, so every tile edge lies on the embedding pixel grid
    tile_size_m: int = Field(strict=True, gt=0, multiple_of=EMBEDDING_PIXEL_SIZE_M)


class GridParams(BaseModel):
    """The grid baseline of square cells with mean embeddings."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    # Whole pixels only, so every cell edge lies on the embedding pixel grid
    cell_size_m: int = Field(strict=True, gt=0, multiple_of=EMBEDDING_PIXEL_SIZE_M)


class Params(BaseModel):
    """All tuned parameters, one section per pipeline stage."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    areas: list[str] = Field(min_length=1)
    export: ExportParams
    grid: GridParams

    @model_validator(mode="after")
    def check_cells_fit_into_tiles(self) -> Self:
        """Check that whole grid cells fill a tile, so no cell crosses a tile edge."""
        if self.export.tile_size_m % self.grid.cell_size_m != 0:
            raise ValueError(
                "export.tile_size_m must be a multiple of grid.cell_size_m"
            )
        return self


def load_params(params_yaml_path: Path = PARAMS_PATH) -> Params:
    """Read and validate params.yaml (by default the one in the project root)."""
    data: Any = yaml.safe_load(stream=params_yaml_path.read_text(encoding="utf-8"))
    return Params.model_validate(obj=data)
