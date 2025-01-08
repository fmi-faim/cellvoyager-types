from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_pascal
from typing import Literal


class Base(BaseModel):
    model_config = ConfigDict(
        alias_generator=to_pascal,
    )


class MeasurementRecordBase(Base):
    time: str
    column: int
    row: int
    time_point: int
    timeline_index: int
    x: float
    y: float
    value: str


class ImageMeasurementRecord(MeasurementRecordBase):
    type: Literal["IMG"]
    field_index: int
    tile_x_index: int | None = None
    tile_y_index: int | None = None
    z_index: int
    action_index: int
    action: str
    z: float
    ch: int


class ErrorMeasurementRecord(MeasurementRecordBase):
    type: Literal["ERR"]


class MeasurementData(Base):
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


class WellPlate(Base):
    version: Literal["1.0"]
    name: str
    product_i_d: str
    usage: str
    density_unit: str
    columns: int
    rows: int
    description: str


class TargetWell(Base):
    column: int
    row: int
    value: bool


class WellSequence(Base):
    is_selected: bool
    target_well: TargetWell


class Point(Base):
    x: int
    y: int


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


class ActionList(Base):
    run_mode: str
    action_acquire_3_d: ActionAcquire3D


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


class LightSource(Base):
    name: str
    type: str
    wave_length: int
    power: int


class LightSourceList(Base):
    use_calibrated_laser_power: bool
    light_source: list[LightSource]


class Channel(Base):
    ch: int
    target: int
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
    c_s_u_i_d: int
    pinhole_diameter: int
    kind: str
    camera_type: str
    input_level: int
    fluorophore: str
    light_source_name: str


class ChannelList(Base):
    channel: list[Channel]


class MeasurementSetting(Base):
    version: Literal["1.0"]
    product_i_d: str
    application: str
    columns: int
    rows: int
    timelapse: Timelapse
    light_source_list: LightSourceList
    channel_list: ChannelList


class CellVoyagerAcquisition(Base):
    well_plate: WellPlate
    measurement_data: MeasurementData
    measurement_detail: MeasurementDetail
    measurement_setting: MeasurementSetting
