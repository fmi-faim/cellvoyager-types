from typing import Annotated, Literal, Optional, Any

from pydantic import BaseModel, ConfigDict, Field, DirectoryPath, field_validator
from pydantic.alias_generators import to_pascal


class Base(BaseModel):
    model_config = ConfigDict(
        alias_generator=to_pascal,
        extra="forbid",
    )


class MeasurementRecordBase(Base):
    time: str
    column: int
    row: int
    field_index: int
    time_point: int
    timeline_index: int
    x: float
    y: float
    value: str


class ImageMeasurementRecord(MeasurementRecordBase):
    type: Literal["IMG"]
    partial_tile_index: int | None = None
    tile_x_index: int | None = None
    tile_y_index: int | None = None
    z_index: int
    z_image_processing: str | None = None
    z_top: float | None = None
    z_bottom: float | None = None
    action_index: int
    action: str
    z: float
    ch: int


class ErrorMeasurementRecord(MeasurementRecordBase):
    type: Literal["IMG","ERR"]


class MeasurementData(Base):
    xmlns: Annotated[dict, Field(alias="xmlns")]
    version: Literal["1.0"]
    measurement_record: list[ImageMeasurementRecord | ErrorMeasurementRecord] | None = (
        None
    )


class MeasurementSamplePlate(Base):
    name: str
    well_plate_file_name: str
    well_plate_product_file_name: str


class MeasurementChannel(Base):
    ch: int
    horizontal_pixel_dimension: float
    vertical_pixel_dimension: float
    camera_number: int
    input_bit_depth: int
    input_level: int
    horizontal_pixels: int
    vertical_pixels: int
    filter_wheel_position: int
    filter_position: int
    shading_correction_source: str
    objective_magnification_ratio: float
    original_horizontal_pixels: int
    original_vertical_pixels: int


class MeasurementDetail(Base):
    xmlns: Annotated[dict, Field(alias="xmlns")]
    version: Literal["1.0"]
    operator_name: str
    title: str
    application: str
    begin_time: str
    end_time: str
    measurement_setting_file_name: str
    column_count: int
    row_count: int
    time_point_count: int
    field_count: int
    z_count: int
    target_system: str
    release_number: str
    status: str
    measurement_sample_plate: MeasurementSamplePlate
    measurement_channel: list[MeasurementChannel]


class WellPlate(Base):
    xmlns: Annotated[dict, Field(alias="xmlns")]
    version: Literal["1.0"]
    name: str
    product_i_d: str
    usage: str
    density_unit: str
    columns: int
    rows: int
    description: str


class TargetWell(Base):
    column: float
    row: float
    value: bool


class WellSequence(Base):
    is_selected: bool = Field(alias='IsSelected')
    target_well: Optional[list[TargetWell]] = Field(
        default=None, alias='TargetWell'
    )

    @field_validator('target_well', mode='before')
    @classmethod
    def _ensure_list(cls, v: Any):
        """Convert single dict to list containing that dict, or handle None."""
        if v is None:
            return None
        if isinstance(v, dict):
            return [v]
        if isinstance(v, list):
            return v
        raise ValueError(f'Expected dict or list, got {type(v)}')


class Point(Base):
    x: float = Field(alias="X")
    y: float = Field(alias="Y")


class FixedPosition(Base):
    is_proportional: bool
    point: list[Point] = Field(alias="Point")

    @field_validator("point", mode="before")
    @classmethod
    def _ensure_list(cls, v: Any):
        if v is None or isinstance(v, list):
            return v
        if isinstance(v, dict):
            return [v]
        raise TypeError(f"Expected dict, list or None, got {type(v).__name__}")


class TiledArea(Base):
    start_point_x: float = Field(alias="StartPointX")
    start_point_y: float = Field(alias="StartPointY")
    end_point_x: float = Field(alias="EndPointX")
    end_point_y: float = Field(alias="EndPointY")


class PartialTiledPosition(Base):
    overlapping_pixels: int = Field(alias="OverlappingPixels")
    scan_method: Literal["Raster","Tile"] = Field(alias="ScanMethod")
    fill: str = Field(alias="Fill")
    tiled_area: TiledArea = Field(alias="TiledArea")


class PointSequence(Base):
    method: Literal["FixedPosition","PartialTiledPosition"] = Field(alias="Method")
    fixed_position: FixedPosition | None = None
    partial_tiled_position: PartialTiledPosition | None = None


class LiveOption(Base):
    period: str = Field(alias="Period")
    interval: str = Field(alias="Interval")
    kind: str = Field(alias="Kind")
    perform_af: str = Field(alias="PerformAF")


class _ActionAcquireBase(Base):
    """Fields shared by all action-acquire structures."""
    x_offset: str = Field(alias="XOffset")
    y_offset: str = Field(alias="YOffset")


class ActionAcquire3D(_ActionAcquireBase):
    af_shift_base: str = Field(alias="AFShiftBase")
    top_distance: str = Field(alias="TopDistance")
    bottom_distance: str = Field(alias="BottomDistance")
    slice_length: str = Field(alias="SliceLength")
    use_soft_focus: str = Field(alias="UseSoftFocus")
    ch: str | list[str] = Field(alias="Ch")
    image_processing: Optional[str] = Field(alias="ImageProcessing", default=None)

    @field_validator("ch", mode="before")
    @classmethod
    def _ensure_list_or_str(cls, v):
        """Handle both single string and list of strings for ch field."""
        if isinstance(v, (str, list)):
            return v
        raise TypeError(f"Expected string or list, got {type(v).__name__}")


class ActionAcquireBF3D(_ActionAcquireBase):
    af_shift_base: str = Field(alias="AFShiftBase")
    top_distance: str = Field(alias="TopDistance")
    bottom_distance: str = Field(alias="BottomDistance")
    slice_length: str = Field(alias="SliceLength")
    ch: str = Field(alias="Ch")


class ActionAcquireBF(_ActionAcquireBase):
    z_offset: str = Field(alias="ZOffset")
    live_option: Optional[LiveOption] = Field(alias="LiveOption", default=None)
    ch: str = Field(alias="Ch")


class ActionAcquire(_ActionAcquireBase):
    z_offset: str = Field(alias="ZOffset")
    ignore_soft_focus: Optional[str] = Field(alias="IgnoreSoftFocus", default=None)
    connected_action: Optional[str] = Field(alias="ConnectedAction", default=None)
    live_option: Optional[LiveOption] = Field(alias="LiveOption", default=None)
    ch: str = Field(alias="Ch")


class ActionList(Base):
    run_mode: str = Field(alias="RunMode")
    a_f_search: Optional[str] = Field(alias="AFSearch", default=None)

    action_acquire: Optional[list[ActionAcquire]] = Field(default=None, alias="ActionAcquire")
    action_acquire_3_d: Optional[list[ActionAcquire3D]] = Field(default=None, alias="ActionAcquire3D")
    action_acquire_bf: Optional[list[ActionAcquireBF]] = Field(default=None, alias="ActionAcquireBF")
    action_acquire_bf_3_d: Optional[list[ActionAcquireBF3D]] = Field(default=None, alias="ActionAcquireBF3D")

    @field_validator(
        "action_acquire",
        "action_acquire_3_d",
        "action_acquire_bf",
        "action_acquire_bf_3_d",
        mode="before",
    )
    def _ensure_list(cls, v):
        if v is None or isinstance(v, list):
            return v
        if isinstance(v, dict):
            return [v]
        raise TypeError(f"Expected dict, list or None, got {type(v).__name__}")


class Timeline(Base):
    name: str
    initial_time: int
    period: int
    interval: int
    expected_time: int
    color: str
    override_expected_time: bool
    well_sequence: WellSequence
    point_sequence: PointSequence
    action_list: ActionList


class Timelapse(Base):
    timeline: list[Timeline]

    @field_validator('timeline', mode='before')
    def _ensure_list(cls, v):
        """Convert single dict to list containing that dict"""
        if isinstance(v, dict):
            return [v]
        if isinstance(v, list):
            return v
        raise ValueError(f'Expected dict or list, got {type(v)}')


class LightSource(Base):
    name: str
    type: str
    wave_length: int
    power: int


class LightSourceList(Base):
    use_calibrated_laser_power: bool | None = None
    light_source: list[LightSource]


class Channel(Base):
    ch: int
    target: str
    objective_i_d: str
    objective: str
    magnification: int
    method_i_d: int
    method: str
    filter_i_d: int
    acquisition: str
    exposure_time: int
    binning: int
    color: str
    min_level: float
    max_level: float
    c_s_u_i_d: int | None = None
    pinhole_diameter: int | None = None
    andor_parameter_i_d: int | None = None
    andor_parameter: str | None = None
    kind: str
    camera_type: str
    input_level: int
    fluorophore: str
    light_source_name: str | list[str]


class ChannelList(Base):
    channel: list[Channel]


class MeasurementSetting(Base):
    xmlns: Annotated[dict, Field(alias="xmlns")]
    version: Literal["1.0"]
    product_i_d: str
    application: str
    columns: int
    rows: int
    timelapse: Timelapse
    light_source_list: LightSourceList
    channel_list: ChannelList


class CellVoyagerAcquisition(Base):
    parent: Annotated[DirectoryPath, Field(alias="parent")]
    well_plate: WellPlate
    measurement_data: MeasurementData
    measurement_detail: MeasurementDetail
    measurement_setting: MeasurementSetting

    def _no_measurement_records(self):
        ValueError("No measurement records found in dataset.")

    def get_wells(self) -> list[tuple[int, int]]:
        if self.measurement_data.measurement_record:
            return list(dict.fromkeys((r.row, r.column) for r in self.get_image_measurement_records()))
        raise(self._no_measurement_records())

    def get_wells_dict(self) -> dict[str, tuple[int, int]]:
        if self.measurement_data.measurement_record:
            letters = "ABCDEFGHIJKLMNOP"
            return {
                f"{letters[r.row-1]}{r.column:02}": (r.row, r.column)
                for r in self.get_image_measurement_records()
            }
        raise(self._no_measurement_records())

    def get_fields(self) -> list[int]:
        if self.measurement_data.measurement_record:
            return list(dict.fromkeys(r.field_index for r in self.get_image_measurement_records()))
        raise(self._no_measurement_records())

    def get_channels(self) -> list[int]:
        if self.measurement_data.measurement_record:
            return list(dict.fromkeys(r.ch for r in self.get_image_measurement_records()))
        raise(self._no_measurement_records())

    def get_time_points(self) -> list[int]:
        if self.measurement_data.measurement_record:
            return list(dict.fromkeys(r.time_point for r in self.get_image_measurement_records()))
        raise(self._no_measurement_records())

    def get_z_indices(self) -> list[int]:
        if self.measurement_data.measurement_record:
            return list(dict.fromkeys(r.z_index for r in self.get_image_measurement_records()))
        raise(self._no_measurement_records())

    def get_timeline_indices(self) -> list[int]:
        if self.measurement_data.measurement_record:
            return list(dict.fromkeys(r.timeline_index for r in self.get_image_measurement_records()))
        raise(self._no_measurement_records())

    def get_action_indices(self) -> list[int]:
        if self.measurement_data.measurement_record:
            return list(dict.fromkeys(r.action_index for r in self.get_image_measurement_records()))
        raise(self._no_measurement_records())

    def get_image_measurement_records(self) -> list[ImageMeasurementRecord]:
        if self.measurement_data.measurement_record:
            return [record for record in self.measurement_data.measurement_record if isinstance(record, ImageMeasurementRecord)]
        raise(self._no_measurement_records())

    def to_dataarray(
            self,
            *,
            columns: list[int] | None = None,
            rows: list[int] | None = None,
            fields: list[int] | None = None,
            channels: list[int] | None = None,
            z_indices: list[int] | None = None,
        ):
        from src.cellvoyager_types._xarray import HAS_XARRAY
        if HAS_XARRAY:
            from src.cellvoyager_types._xarray import dataarray_from_metadata
        else:
            raise ValueError("Dependencies for data array creation not found.")
        if not self.measurement_data.measurement_record:
            raise ValueError("No measurement records found in dataset.")
        image_records = self.get_image_measurement_records()
        if columns is not None:
            image_records = [record for record in image_records if record.column in columns]
        if rows is not None:
            image_records = [record for record in image_records if record.row in rows]
        if fields is not None:
            image_records = [record for record in image_records if record.field_index in fields]
        if channels is not None:
            image_records = [record for record in image_records if record.ch in channels]
        if z_indices is not None:
            image_records = [record for record in image_records if record.z_index in z_indices]
        if len(image_records) == 0:
            msg = f"""
                No image records found for the specified subset.
                Available rows: {set(record.row for record in self.get_image_measurement_records())}
                Available columns: {set(record.column for record in self.get_image_measurement_records())}
                Available fields: {set(record.field_index for record in self.get_image_measurement_records())}
                Available channels: {set(record.ch for record in self.get_image_measurement_records())}
                """
            raise ValueError(msg)

        return dataarray_from_metadata(
            parent_folder=self.parent,
            image_records=image_records,
            detail=self.measurement_detail,
        )
