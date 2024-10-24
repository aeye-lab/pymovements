from dataclasses import dataclass
from typing import List, Tuple
import datetime

@dataclass
class MountConfiguration:
    mount_type: str
    head_stabilization: str
    eyes_recorded: str
    short_name: str

@dataclass
class CalibrationMetadata:
    timestamp: str
    num_points: int
    type: str
    tracked_eye: str

@dataclass
class ValidationMetadata:
    timestamp: str
    num_points: int
    tracked_eye: str
    error: str
    validation_score_avg: float
    validation_score_max: float

@dataclass
class BlinkMetadata:
    start_timestamp: float
    stop_timestamp: float
    duration_ms: float
    num_samples: int

@dataclass
class ASCMetadata:
    year: int
    tracked_eye: str
    version_1: str
    version_2: str
    resolution: Tuple[int, int]
    tracking_mode: str
    sampling_rate: float
    file_sample_filter: int
    link_sample_filter: int
    mount_configuration: MountConfiguration
    pupil_data_type: str
    version_number: float
    model: str
    datetime: datetime.datetime
    calibrations: List[CalibrationMetadata]
    validations: List[ValidationMetadata]
    blinks: List[BlinkMetadata]
    data_loss_ratio: float
    data_loss_ratio_blinks: float
    total_recording_duration_ms: float

# Initialize metadata_obj with example data
metadata_obj = ASCMetadata(
    year=2024,
    tracked_eye='Right',
    version_1='EYELINK II 1',
    version_2='EYELINK II CL v6.14 Mar 6 2020 (EyeLink Portable Duo)',
    resolution=(1276, 917),
    tracking_mode='CR',
    sampling_rate=2000.0,
    file_sample_filter=0,
    link_sample_filter=0,
    mount_configuration=MountConfiguration(
        mount_type='Desktop',
        head_stabilization='stabilized',
        eyes_recorded='binocular / monocular',
        short_name='BTABLER'
    ),
    pupil_data_type='AREA',
    version_number=6.14,
    model='EyeLink Portable Duo',
    datetime=datetime.datetime(2024, 6, 19, 17, 49, 30),
    calibrations=[
        CalibrationMetadata(timestamp='623940.0', num_points=9, type='P-CR', tracked_eye='RIGHT')
    ],
    validations=[
        ValidationMetadata(
            timestamp='643476.0', num_points=9, tracked_eye='RIGHT',
            error='GOOD ERROR', validation_score_avg=0.41, validation_score_max=0.63
        )
    ],
    blinks=[
        BlinkMetadata(start_timestamp=715946.0, stop_timestamp=717035.0, duration_ms=1089.0, num_samples=2181)
    ],
    data_loss_ratio=0.04547720970203655,
    data_loss_ratio_blinks=0.044510408788512146,
    total_recording_duration_ms=1610466.0
)

# Accessing calibrations
#print(metadata_obj)
print(metadata_obj.calibrations)
print(metadata_obj.mount_configuration)