from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from earth_similarity_explorer.config import AreaConfig, load_area_config
from earth_similarity_explorer.constants import LAST_EMBEDDING_YEAR

AREAS_DIR: Path = Path(__file__).parents[1] / "areas"

VALID_CONFIG: dict[str, str | int] = {
    "id": "test-area",
    "name": "Test area",
    "year": 2024,
    "boundary": "geometries/test-area.geojson",
}


# HELPER FUNCTIONS
def write_config(directory: Path, **changes: object) -> Path:
    """Write the valid config, with optional changes, as test-area.yaml."""
    path: Path = directory / "test-area.yaml"
    path.write_text(data=yaml.safe_dump(data=VALID_CONFIG | changes), encoding="utf-8")
    return path


def test_valid_config_loads(tmp_path: Path) -> None:
    """The config loads correctly."""
    config: AreaConfig = load_area_config(
        config_yaml_path=write_config(directory=tmp_path)
    )

    assert config.id == "test-area"
    assert config.year == 2024
    assert config.boundary == Path("geometries/test-area.geojson")


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
    """File name must match the id"""
    path: Path = write_config(directory=tmp_path).rename(
        target=tmp_path / "other-area.yaml"
    )

    with pytest.raises(
        expected_exception=ValueError, match="does not match the file name"
    ):
        load_area_config(config_yaml_path=path)


@pytest.mark.parametrize(
    argnames="path",
    argvalues=sorted(AREAS_DIR.glob(pattern="*.yaml")),
    ids=lambda p: p.stem,
)
def test_repository_area_configs_are_valid(path: Path) -> None:
    """All config.yaml in the /areas folder load correctly."""
    load_area_config(config_yaml_path=path)
