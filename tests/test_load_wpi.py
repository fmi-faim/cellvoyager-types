import pytest
from pathlib import Path

from cellvoyager_types import load_wpi


@pytest.fixture
def wpi_path() -> Path:
    return Path("tests/resources/20240926-Illumination-QC-60xW.wpi")


def test_load_wpi(wpi_path: Path) -> None:
    assert wpi_path.exists()
    metadata = load_wpi(wpi_path)

    assert metadata

    assert metadata.well_plate
    assert metadata.measurement_data
    assert metadata.measurement_detail
    assert metadata.measurement_setting

    assert isinstance(metadata.measurement_data.measurement_record, list)
    assert len(metadata.measurement_data.measurement_record) == 1984
    assert len(metadata.measurement_setting.channel_list.channel) == 4


def test_missing_wpi_file() -> None:
    with pytest.raises(FileNotFoundError):
        load_wpi(Path("tests/resources/missing.wpi"))
