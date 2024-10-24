from dataclasses import dataclass
from dataclasses import asdict, is_dataclass
from typing import List, Tuple, Dict, Literal, Any
import datetime

@dataclass
class MountConfiguration:
    """
    Represents the configuration of the mount used for eye tracking.

    Attributes:
        mount_type (str): Type of the mount configuration used (e.g., "Desktop", "Portable").
        head_stabilization (str): Method of head stabilization used during tracking (e.g., "stabilized", "unstabilized").
        eyes_recorded (str): Indicates whether both eyes or a single eye were recorded (e.g., "binocular / monocular").
        short_name (str): A concise name or identifier for the mount configuration (e.g., "BTABLER").
    """
    mount_type: str
    head_stabilization: str
    eyes_recorded: str
    short_name: str

@dataclass
class CalibrationMetadata:
    """
    Represents metadata related to calibration events.

    Attributes:
        timestamp (str):  The time at which the calibration occurred, currently it is stored as a string but should be a float.
        num_points (int): The number of calibration points used during the calibration process.
        type (str): Type of calibration performed (e.g., "P-CR").
        tracked_eye (str): Specifies which eye was tracked during calibration (e.g., "RIGHT").
    """
    timestamp: str
    num_points: int
    type: str
    tracked_eye: str

@dataclass
class ValidationMetadata:
    """
    Represents metadata related to validation events.

    Attributes:
        timestamp (str): The time at which the validation occurred, currently it is stored as a string but should be a float..
        num_points (int): The number of validation points that were checked.
        tracked_eye (str): Specifies which eye was validated during the validation process (e.g., "RIGHT").
        error (str): Describes the quality of the validation (e.g., "GOOD ERROR").
        validation_score_avg (float): Average score obtained during the validation process.
        validation_score_max (float): Maximum score obtained during validation.
    """
    timestamp: str
    num_points: int
    tracked_eye: str
    error: str
    validation_score_avg: float
    validation_score_max: float

@dataclass
class BlinkMetadata:
    """
    Represents metadata related to blink events.

    Attributes:
        start_timestamp (float): The timestamp (in milliseconds) at which the blink started.
        stop_timestamp (float): The timestamp (in milliseconds) at which the blink stopped.
        duration_ms (float): Duration of the blink in milliseconds.
        num_samples (int): The number of samples recorded during the blink event.
    """
    start_timestamp: float
    stop_timestamp: float
    duration_ms: float
    num_samples: int

@dataclass
class ASCMetadata:
    """
    Represents the overall metadata for the eye tracking session.

    Attributes:
        year (int): The year during which the data was recorded (e.g., 2024).
        tracked_eye (Literal["Right", "Left"]): Indicates which eye was tracked (either "Right" or "Left").
        version_1 (str): Version information for the tracking software or hardware (e.g., "EYELINK II 1").
        version_2 (str): Additional version details (e.g., "EYELINK II CL v6.14 Mar 6 2020 (EyeLink Portable Duo)").
        resolution (Tuple[int, int]): The resolution of the recorded video or data, represented as a tuple of (width, height) in pixels (e.g., (1276, 917)).
        tracking_mode (str): The mode of tracking used during the recording session (e.g., "CR").
        sampling_rate (float): The frequency at which data points were recorded, in Hz (e.g., 2000.0).
        file_sample_filter (int): A filter applied to the samples in the recorded file (e.g., 0 for no filtering).
        link_sample_filter (int): The level of filtering applied to samples (0 = off, 1 = standard, 2 = extra)
        mount_configuration (MountConfiguration): An instance of the MountConfiguration class, containing specific mount configuration details.
        pupil_data_type (str): Type of pupil data collected during the experiment (e.g., "AREA").
        version_number (float): Version number of the tracking hardware or software (e.g., 6.14).
        model (str): The model of the tracking device used (e.g., "EyeLink Portable Duo").
        datetime (datetime.datetime): The date and time when the recording took place, represented as a datetime object (e.g., datetime(2024, 6, 19, 17, 49, 30)).
        calibrations (List[CalibrationMetadata]): A list of CalibrationMetadata instances, each representing a calibration event recorded during the session.
        validations (List[ValidationMetadata]): A list of ValidationMetadata instances, each representing a validation event recorded during the session.
        blinks (List[BlinkMetadata]): A list of BlinkMetadata instances, each representing a blink event recorded during the session.
        data_loss_ratio (float): Ratio of lost data points during the recording session.
        data_loss_ratio_blinks (float): Ratio of lost blink data points during the recording session.
        total_recording_duration_ms (float): Total duration of the recording session in milliseconds.
    """
    year: int
    tracked_eye: Literal["Right", "Left"]
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

# Example instantiation
metadata_obj = ASCMetadata(
    year=2024,
    tracked_eye="Right",
    version_1="EYELINK II 1",
    version_2="EYELINK II CL v6.14 Mar 6 2020 (EyeLink Portable Duo)",
    resolution=(1276, 917),
    tracking_mode="CR",
    sampling_rate=2000.0,
    file_sample_filter=0,
    link_sample_filter=0,
    mount_configuration=MountConfiguration(
        mount_type="Desktop",
        head_stabilization="stabilized",
        eyes_recorded="binocular / monocular",
        short_name="BTABLER"
    ),
    pupil_data_type="AREA",
    version_number=6.14,
    model="EyeLink Portable Duo",
    datetime=datetime.datetime(2024, 6, 19, 17, 49, 30),
    calibrations=[
        CalibrationMetadata(timestamp="623940.0", num_points=9, type="P-CR", tracked_eye="RIGHT")
    ],
    validations=[
        ValidationMetadata(
            timestamp="643476.0", num_points=9, tracked_eye="RIGHT",
            error="GOOD ERROR", validation_score_avg=0.41, validation_score_max=0.63
        )
    ],
    blinks=[
        BlinkMetadata(start_timestamp=715946.0, stop_timestamp=717035.0, duration_ms=1089.0, num_samples=2181)
    ],
    data_loss_ratio=0.04547720970203655,
    data_loss_ratio_blinks=0.044510408788512146,
    total_recording_duration_ms=1610466.0
)

#print(metadata_obj)  
print(metadata_obj.year)
print(metadata_obj.mount_configuration)
print(metadata_obj.validations)

# Print the type of the timestamp attributes from the metadata_obj
print("CalibrationMetadata timestamp type:", type(metadata_obj.calibrations[0].timestamp))
print("ValidationMetadata timestamp type:", type(metadata_obj.validations[0].timestamp))
print("BlinkMetadata start_timestamp type:", type(metadata_obj.blinks[0].start_timestamp))
print("BlinkMetadata stop_timestamp type:", type(metadata_obj.blinks[0].stop_timestamp))

