from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from earth_similarity_explorer.config import load_area_config
from earth_similarity_explorer.params import Params, load_params

REPOSITORY_ROOT: Path = Path(__file__).parents[1]

VALID_PARAMS: dict[str, object] = {"areas": ["test-area"], "grid": {"cell_size_m": 100}}


# TESTS
@pytest.mark.parametrize(
    argnames="changes",
    argvalues=[
        {"grid": {"cell_size_m": 105}},
        {"grid": {"cell_size_m": "100"}},
        {"areas": []},
        {"segmentation": {}},
    ],
    ids=["cell-cuts-pixels", "cell-size-as-string", "no-areas", "unknown-section"],
)
def test_invalid_params_fail(tmp_path: Path, changes: dict[str, object]) -> None:
    """Invalid parameters fail to load."""
    path: Path = tmp_path / "params.yaml"
    path.write_text(data=yaml.safe_dump(data=VALID_PARAMS | changes), encoding="utf-8")

    with pytest.raises(expected_exception=ValidationError):
        load_params(params_yaml_path=path)


def test_repository_params_are_valid() -> None:
    """The real params.yaml loads, and every area it lists has a valid config."""
    params: Params = load_params(params_yaml_path=REPOSITORY_ROOT / "params.yaml")

    for area_id in params.areas:
        load_area_config(config_yaml_path=REPOSITORY_ROOT / "areas" / f"{area_id}.yaml")
