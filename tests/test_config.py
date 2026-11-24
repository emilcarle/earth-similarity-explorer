"""Tests for the area config."""

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from earth_similarity_explorer.config import AreaConfig, load_area_config
from earth_similarity_explorer.settings import LAST_EMBEDDING_YEAR

VALID_CONFIG: dict[str, str | int] = {
    "id": "test-area",
    "name": "Test area",
    "year": 2024,
    "boundary": "data/geometries/test-area.geojson",
}


# HELPER FUNCTIONS
def write_config(directory: Path, **changes: object) -> Path:
    """Write the valid config, with optional changes, as test-area.yaml."""
    path: Path = directory / "test-area.yaml"
    path.write_text(data=yaml.safe_dump(data=VALID_CONFIG | changes), encoding="utf-8")
    return path


# TESTS
def test_valid_config_loads(tmp_path: Path) -> None:
    """A valid config loads with its values."""
    config: AreaConfig = load_area_config(
        config_yaml_path=write_config(directory=tmp_path)
    )

    assert config.id == "test-area"
    assert config.year == 2024
    assert config.boundary == Path("data/geometries/test-area.geojson")


@pytest.mark.parametrize(
    argnames=("field", "value"),
    argvalues=[
        ("id", "Test_Area"),
        ("year", LAST_EMBEDDING_YEAR + 1),
        ("year", "2024"),
        ("yaer", 2024),
    ],
    ids=["id-format", "year-out-of-range", "year-as-string", "unknown-key"],
)
def test_invalid_config_fails(tmp_path: Path, field: str, value: object) -> None:
    """An invalid config fails to load and produces a ValidationError."""
    with pytest.raises(expected_exception=ValidationError) as exc_info:
        load_area_config(
            config_yaml_path=write_config(directory=tmp_path, **{field: value})
        )
    # Check that there is exactly one error, on the changed field
    assert [error["loc"] for error in exc_info.value.errors()] == [(field,)]


def test_file_name_must_match_id(tmp_path: Path) -> None:
    """A config whose file name differs from its id fails to load."""
    path: Path = write_config(directory=tmp_path).rename(
        target=tmp_path / "other-area.yaml"
    )

    with pytest.raises(
        expected_exception=ValueError, match="does not match the file name"
    ):
        load_area_config(config_yaml_path=path)
