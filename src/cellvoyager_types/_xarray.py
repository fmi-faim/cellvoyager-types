try:
    import dask.array as da
    import xarray as xr
    from tifffile import imread
    from collections import defaultdict
    HAS_XARRAY = True
except ImportError:
    HAS_XARRAY = False

from pathlib import Path
from cellvoyager_types._metadata import ImageMeasurementRecord

def dataarray_from_metadata(parent_folder: Path, image_records: list[ImageMeasurementRecord]) -> "xr.DataArray":
    """
    Organizes images and loads them into a combined DataArray.

    Args:
        image_records: List of ImageMeasurementRecord objects

    Returns:
        xr.DataArray: Combined dataset with proper coordinates
    """
    # Group images by field_index and channel
    grouped_images = defaultdict(list)
    for img in image_records:
        key = (img.column, img.row, img.field_index, img.ch)
        grouped_images[key].append(img)

    # Create Datasets with all dimensions
    datasets = []
    for (col, row, field_idx, ch), images in grouped_images.items():
        # Sort by z_index
        sorted_images: list[ImageMeasurementRecord] = sorted(images, key=lambda x: x.z_index)

        # Load images as dask arrays via zarr
        chunks = (-1, -1)
        dask_arrays = []
        for img in sorted_images:
            with imread(parent_folder / img.value, aszarr=True) as store:
                data = da.from_zarr(store, chunks=chunks)
                dask_arrays.append(data)

        # Stack arrays along a new first axis
        stacked = da.stack(dask_arrays).rechunk(chunks=(len(dask_arrays), *chunks))

        # Create DataArray with all coordinates at once
        coords = {
            'row': [row],
            'column': [col],
            'field': [field_idx],
            'channel': [ch],
            'z': [img.z_index for img in sorted_images],  # TODO z_index is 1-based currently
            'y': range(stacked.shape[1]),
            'x': range(stacked.shape[2])
        }

        data_array = xr.DataArray(
            # data=stacked.reshape(1, 1, 1, 1, *stacked.shape),
            data=stacked[None, None, None, None, ...], # Add singleton dimensions for col, row, field, channel
            dims=['row', 'column', 'field', 'channel', 'z', 'y', 'x'],
            coords=coords
        )

        # Wrap DataArray in a Dataset
        datasets.append(xr.Dataset({'intensity': data_array}))

    # Combine all datasets and extract the intensity DataArray
    combined = xr.combine_by_coords(datasets)['intensity']
    return combined
