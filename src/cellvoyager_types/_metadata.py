from pydantic import BaseModel, ConfigDict, DirectoryPath, Field, field_validator
from pydantic.alias_generators import to_pascal
from typing import Annotated, Literal


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
    type: Literal["ERR"]


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
    is_selected: bool
    target_well: list[TargetWell]

    @field_validator('target_well', mode='before')
    def _ensure_list(cls, v):
        """Convert single dict to list containing that dict"""
        if isinstance(v, dict):
            return [v]
        if isinstance(v, list):
            return v
        raise ValueError(f'Expected dict or list, got {type(v)}')


class Point(Base):
    x: float
    y: float


class FixedPosition(Base):
    is_proportional: bool
    point: list[Point]


class PointSequence(Base):
    method: str
    fixed_position: FixedPosition


class ActionAcquire3D(Base):
    x_offset: int
    y_offset: int
    a_f_shift_base: int
    top_distance: int
    bottom_distance: int
    slice_length: int
    use_soft_focus: bool
    ch: int
    image_processing: str | None = None


class ActionList(Base):
    run_mode: str
    a_f_search: str | None = None
    action_acquire_3_d: list[ActionAcquire3D]

    @field_validator('action_acquire_3_d', mode='before')
    def _ensure_list(cls, v):
        """Convert single dict to list containing that dict"""
        if isinstance(v, dict):
            return [v]
        if isinstance(v, list):
            return v
        raise ValueError(f'Expected dict or list, got {type(v)}')


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
    light_source_name: str


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
            return list(dict.fromkeys((r.row, r.column) for r in self.measurement_data.measurement_record))
        raise(self._no_measurement_records())

    def get_wells_dict(self) -> dict[str, tuple[int, int]]:
        if self.measurement_data.measurement_record:
            letters = "ABCDEFGHIJKLMNOP"
            return {
                f"{letters[r.row-1]}{r.column:02}": (r.row, r.column)
                for r in self.measurement_data.measurement_record
            }
        raise(self._no_measurement_records())

    def get_fields(self) -> list[int]:
        if self.measurement_data.measurement_record:
            return list(dict.fromkeys(r.field_index for r in self.measurement_data.measurement_record))
        raise(self._no_measurement_records())

    def get_channels(self) -> list[int]:
        if self.measurement_data.measurement_record:
            return list(dict.fromkeys(r.ch for r in self.measurement_data.measurement_record))
        raise(self._no_measurement_records())

    def get_time_points(self) -> list[int]:
        if self.measurement_data.measurement_record:
            return list(dict.fromkeys(r.time_point for r in self.measurement_data.measurement_record))
        raise(self._no_measurement_records())

    def get_z_indices(self) -> list[int]:
        if self.measurement_data.measurement_record:
            return list(dict.fromkeys(r.z_index for r in self.measurement_data.measurement_record))
        raise(self._no_measurement_records())

    def get_timeline_indices(self) -> list[int]:
        if self.measurement_data.measurement_record:
            return list(dict.fromkeys(r.timeline_index for r in self.measurement_data.measurement_record))
        raise(self._no_measurement_records())

    def get_action_indices(self) -> list[int]:
        if self.measurement_data.measurement_record:
            return list(dict.fromkeys(r.action_index for r in self.measurement_data.measurement_record))
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
        from cellvoyager_types._xarray import HAS_XARRAY
        if HAS_XARRAY:
            from cellvoyager_types._xarray import dataarray_from_metadata
        else:
            raise ValueError("Dependencies for data array creation not found.")
        if not self.measurement_data.measurement_record:
            raise ValueError("No measurement records found in dataset.")
        image_records = [record for record in self.measurement_data.measurement_record if isinstance(record, ImageMeasurementRecord)]
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
                Available rows: {set(record.row for record in self.measurement_data.measurement_record)}
                Available columns: {set(record.column for record in self.measurement_data.measurement_record)}
                Available fields: {set(record.field_index for record in self.measurement_data.measurement_record)}
                Available channels: {set(record.ch for record in self.measurement_data.measurement_record)}
                """
            raise ValueError(msg)

        return dataarray_from_metadata(
            parent_folder=self.parent,
            image_records=image_records,
            detail=self.measurement_detail,
        )
