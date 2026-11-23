"""Checks on the project's own data files: params.yaml, area configs and boundaries."""

from pathlib import Path

import pytest

from earth_similarity_explorer.boundary import load_boundary
from earth_similarity_explorer.config import AreaConfig, load_area_config
from earth_similarity_explorer.params import Params, load_params

REPOSITORY_ROOT: Path = Path(__file__).parents[1]
AREAS_DIR: Path = REPOSITORY_ROOT / "areas"


# TESTS
@pytest.mark.parametrize(
    argnames="config_path",
    argvalues=sorted(AREAS_DIR.glob(pattern="*.yaml")),
    ids=lambda p: p.stem,
)
def test_area_config_and_boundary_are_valid(config_path: Path) -> None:
    """Every area config in areas/ is valid and points to a valid boundary file."""
    config: AreaConfig = load_area_config(config_yaml_path=config_path)

    load_boundary(path=REPOSITORY_ROOT / config.boundary)
    # dvc.yaml finds the boundary by this name, not by reading the config
    assert config.boundary == Path("data/geometries") / f"{config.id}.geojson"


def test_params_are_valid_and_list_existing_areas() -> None:
    """params.yaml is valid, and every area it lists has a config in areas/."""
    params: Params = load_params(params_yaml_path=REPOSITORY_ROOT / "params.yaml")

    for area_id in params.areas:
        assert (AREAS_DIR / f"{area_id}.yaml").is_file(), f"no config for {area_id!r}"
