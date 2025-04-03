import pytest
from pathlib import Path
from cellvoyager_types import load_wpi
from cellvoyager_types._xarray import HAS_XARRAY


@pytest.fixture
def cv_acquisition():
    return load_wpi(Path("tests/resources/CV8000-Minimal-DataSet-2C-3W-4S-FP2-stack/CV8000-Minimal-DataSet-2C-3W-4S-FP2-stack.wpi"))


def test_xarray_default(cv_acquisition):
    if HAS_XARRAY:
        array = cv_acquisition.to_dataarray().compute()
        assert array.dims == ("row", "column", "field", "channel", "z", "y", "x")
        assert array.shape == (3, 2, 4, 2, 4, 2000, 2000)
        assert cv_acquisition.to_dataarray().data.chunksize == (1, 1, 1, 1, 1, 2000, 2000)
    else:
        with pytest.raises(ValueError):
            cv_acquisition.to_dataarray()


def test_xarray_subset(cv_acquisition):
    if HAS_XARRAY:
        array = cv_acquisition.to_dataarray(
            rows=[4],
            columns=[8],
            fields=[1, 3, 4],
            channels=[1, 2],
            z_indices=[2, 3],
        ).compute()
        assert array.dims == ("row", "column", "field", "channel", "z", "y", "x")
        assert array.shape == (1, 1, 3, 2, 2, 2000, 2000)
        squeezed = array.squeeze()
        assert squeezed.dims == ("field", "channel", "z", "y", "x")
        assert squeezed.shape == (3, 2, 2, 2000, 2000)
        assert squeezed.dtype == "uint16"
    else:
        with pytest.raises(ValueError):
            cv_acquisition.to_dataarray()


def test_xarray_subset_downsampled(cv_acquisition):
    if HAS_XARRAY:
        array = cv_acquisition.to_dataarray(
            rows=[4],
            columns=[8],
            fields=[1, 3],
            channels=[1],
            z_indices=[3, 4],
        )
        result = array.coarsen(y=4, x=4).mean().squeeze().compute()
        assert result.shape == (2, 2, 500, 500)
        assert result.dims == ("field", "z", "y", "x")


def test_xarray_invalid(cv_acquisition):
    if HAS_XARRAY:
        with pytest.raises(ValueError, match='No image records found for the specified subset.'):
            cv_acquisition.to_dataarray(
                rows=[9],
                columns=[20],
                fields=[1],
                channels=[3],
            )
    else:
        with pytest.raises(ValueError):
            cv_acquisition.to_dataarray()
