import pytest
from pathlib import Path

from cellvoyager_types import load_wpi


@pytest.mark.parametrize(
    "wpi_path, records, channels",
    [
        pytest.param(
            Path("tests/resources/CV8000-Minimal-DataSet-2C-3W-4S-FP2-stack/CV8000-Minimal-DataSet-2C-3W-4S-FP2-stack.wpi"),
            96,
            4,
            id="CV8000-Minimal-DataSet",
        ),
        pytest.param(
            Path("tests/resources/CV7000-example/20250303-Illumination-QC-20x.wpi"),
            1984,
            6,
            id="CV7000-example",
        ),
        pytest.param(
            Path("tests/resources/CV7000_z_processed/AssayPlate_Cellvis_#384-1.5H-N.wpi"),
            135,
            5,
            id="CV7000_z_processed",
        ),
        pytest.param(
            Path("tests/resources/CV8000_AF_Error/20250403-Illumination-QC-40x.wpi"),
            1864,
            4,
            id="CV8000_AF_Error",
        ),
        pytest.param(
            Path("tests/resources/20240926-Illumination-QC-60xW.wpi"),
            1984,
            4,
            id="CV8000_Illumination-QC",
        ),
    ],
)
def test_load_wpi(wpi_path: Path, records: int, channels: int) -> None:
    assert wpi_path.exists()
    metadata = load_wpi(wpi_path)

    assert metadata
    assert metadata.parent == wpi_path.parent

    assert metadata.well_plate
    assert metadata.measurement_data
    assert metadata.measurement_detail
    assert metadata.measurement_setting

    assert isinstance(metadata.measurement_data.measurement_record, list)
    assert len(metadata.measurement_data.measurement_record) == records
    assert len(metadata.measurement_setting.channel_list.channel) == channels

    assert metadata.measurement_detail.measurement_channel
    # assert len(metadata.measurement_detail.measurement_channel) == channels


def test_missing_wpi_file() -> None:
    with pytest.raises(FileNotFoundError):
        load_wpi(Path("tests/resources/missing.wpi"))
