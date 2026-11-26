"""Tests for the pipeline parameters."""

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from earth_similarity_explorer.params import load_params

VALID_PARAMS: dict[str, object] = {
    "areas": ["test-area"],
    "export": {"tile_size_m": 20000},
    "grid": {"cell_size_m": 100},
}


# TESTS
@pytest.mark.parametrize(
    argnames="changes",
    argvalues=[
        {"grid": {"cell_size_m": 105}},
        {"grid": {"cell_size_m": "100"}},
        {"export": {"tile_size_m": 20005}},
        # 20 km is not a multiple of 300 m, so a cell would cross a tile edge
        {"grid": {"cell_size_m": 300}},
        {"areas": []},
        {"segmentation": {}},
    ],
    ids=[
        "cell-cuts-pixels",
        "cell-size-as-string",
        "tile-cuts-pixels",
        "cell-crosses-tiles",
        "no-areas",
        "unknown-section",
    ],
)
def test_invalid_params_fail(tmp_path: Path, changes: dict[str, object]) -> None:
    """Invalid parameters fail to load."""
    path: Path = tmp_path / "params.yaml"
    path.write_text(data=yaml.safe_dump(data=VALID_PARAMS | changes), encoding="utf-8")

    with pytest.raises(expected_exception=ValidationError):
        load_params(params_yaml_path=path)
