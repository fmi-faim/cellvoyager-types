try:
    import dask.array as da
    import xarray as xr
    from tifffile import imread
    from collections import defaultdict
    from functools import partial
    HAS_XARRAY = True
except ImportError:
    HAS_XARRAY = False

from pathlib import Path
from cellvoyager_types._metadata import ImageMeasurementRecord, MeasurementDetail

def dataarray_from_metadata(parent_folder: Path, image_records: list[ImageMeasurementRecord], detail: MeasurementDetail) -> "xr.DataArray":
    """
    Organizes images and loads them into a combined DataArray.

    Args:
        image_records: List of ImageMeasurementRecord objects

    Returns:
        xr.DataArray: Combined dataset with proper coordinates
    """
    # Group images by well, field_index and channel
    grouped_images = defaultdict(list)
    for img in image_records:
        key = (img.column, img.row, img.field_index, img.ch)
        grouped_images[key].append(img)

    measurement_channels = {ch.ch: ch for ch in detail.measurement_channel}

    def _load_image(_, block_id=None, images=None):
        return imread(parent_folder / images[block_id[0]].value)[None, ...]

    # Create Datasets with all dimensions
    datasets = []
    for (col, row, field_idx, ch), images in grouped_images.items():
        # Sort by z_index
        sorted_images: list[ImageMeasurementRecord] = sorted(images, key=lambda x: x.z_index)

        # Load images as dask arrays
        horizontal_pixels = measurement_channels[ch].horizontal_pixels
        vertical_pixels = measurement_channels[ch].vertical_pixels
        dtype = "uint16" if measurement_channels[ch].input_bit_depth == 16 else "uint8"  # TODO more bit depths
        chunks = (1, vertical_pixels, horizontal_pixels)
        volume = da.zeros((len(sorted_images), vertical_pixels, horizontal_pixels), chunks=chunks, dtype=dtype)
        _func = partial(_load_image, images=sorted_images)
        stacked = da.map_blocks(_func, volume, chunks=chunks, dtype=dtype)

        # Create DataArray with all coordinates at once
        coords = {
            'row': [row],
            'column': [col],
            'field': [field_idx],
            'channel': [ch],
            'z': [img.z_index for img in sorted_images],  # z_index is 1-based
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
