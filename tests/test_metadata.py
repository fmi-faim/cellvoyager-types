import pytest
from pathlib import Path
from cellvoyager_types import load_wpi

@pytest.fixture
def cv8000_acquisition():
    return load_wpi(Path("tests/resources/CV8000-Minimal-DataSet-2C-3W-4S-FP2-stack/CV8000-Minimal-DataSet-2C-3W-4S-FP2-stack.wpi"))

@pytest.fixture
def cv7000_acquisition():
    return load_wpi(Path("tests/resources/CV7000-example/20250303-Illumination-QC-20x.wpi"))

def test_dimensions(cv8000_acquisition):
    assert cv8000_acquisition.get_action_indices() == [1, 2]
    assert cv8000_acquisition.get_channels() == [1, 2]
    assert cv8000_acquisition.get_fields() == [1, 2, 3, 4]
    assert cv8000_acquisition.get_time_points() == [1]
    assert cv8000_acquisition.get_wells() == [(4, 8), (5, 3), (6, 8)]
    assert cv8000_acquisition.get_wells_dict() == {"D08": (4, 8), "E03": (5, 3), "F08": (6, 8)}
    assert cv8000_acquisition.get_z_indices() == [1, 2, 3, 4]
    assert cv8000_acquisition.get_timeline_indices() == [1]

def test_cv7000_acquisition(cv7000_acquisition):
    assert len(cv7000_acquisition.measurement_setting.channel_list.channel) == 6
    assert len(cv7000_acquisition.measurement_detail.measurement_channel) == 6
    assert len(cv7000_acquisition.measurement_data.measurement_record) == 1984
