import pytest
from pathlib import Path
from cellvoyager_types import load_wpi

@pytest.fixture
def cv_acquisition():
    return load_wpi(Path("tests/resources/CV8000-Minimal-DataSet-2C-3W-4S-FP2-stack/CV8000-Minimal-DataSet-2C-3W-4S-FP2-stack.wpi"))

def test_dimensions(cv_acquisition):
    assert cv_acquisition.get_action_indices() == [1, 2]
    assert cv_acquisition.get_channels() == [1, 2]
    assert cv_acquisition.get_fields() == [1, 2, 3, 4]
    assert cv_acquisition.get_time_points() == [1]
    assert cv_acquisition.get_wells() == [(4, 8), (5, 3), (6, 8)]
    assert cv_acquisition.get_wells_dict() == {"D08": (4, 8), "E03": (5, 3), "F08": (6, 8)}
    assert cv_acquisition.get_z_indices() == [1, 2, 3, 4]
    assert cv_acquisition.get_timeline_indices() == [1]
