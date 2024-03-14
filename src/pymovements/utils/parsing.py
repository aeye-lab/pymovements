# Copyright (c) 2023-2024 The pymovements Project Authors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Module for parsing input data."""
from __future__ import annotations

import calendar
import datetime
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

EYE_TRACKING_SAMPLE = re.compile(
    r'(?P<time>(\d+))\s+'
    r'(?P<x_pix>[-]?\d*[.]\d*)\s+'
    r'(?P<y_pix>[-]?\d*[.]\d*)\s+'
    r'(?P<pupil>\d*[.]\d*)\s+'
    r'(?P<dummy>\d*[.]\d*)\s+'
    r'(?P<dots>[A-Za-z.]{3,5})?\s*',
)

EYELINK_META_REGEXES = [
    {'pattern': r'\*\*\s+VERSION:\s+(?P<version_1>.*)\s+'},
    {
        'pattern': r'\*\*\s+DATE:\s+(?P<weekday>[A-Z,a-z]+)\s+(?P<month>[A-Z,a-z]+)'
                   r'\s+(?P<day>\d\d?)\s+(?P<time>\d\d:\d\d:\d\d)\s+(?P<year>\d{4})\s+',
    },
    {'pattern': r'\*\*\s+(?P<version_2>EYELINK.*)'},
    {
        'pattern': r'SAMPLES\s+GAZE\s+(?P<tracked_eye>LEFT|RIGHT)\s+RATE\s+'
                   r'(?P<sampling_rate>[-]?\d*[.]\d*)\s+TRACKING\s+(?P<tracking>\S+)'
                   r'\s+FILTER\s+(?P<filter>\d+)\s+INPUT',
    },
    {
        'pattern': r'MSG\s+\d+\s+DISPLAY_COORDS\s+(?P<resolution>.*)',
    },
]

VALIDATION_REGEX = (
    r'MSG\s+(?P<timestamp>\d+)\s+!CAL\s+VALIDATION\s+HV'
    r'(?P<num_points>\d\d?).*'
    r'(?P<eye_tracked>LEFT|RIGHT)\s+'
    r'(?P<error>\D*)\s+'
    r'(?P<validation_score_avg>\d.\d\d)\s+avg\.\s+'
    r'(?P<validation_score_max>\d.\d\d)\s+max'
)

BLINK_START_REGEX = r'SBLINK\s+(R|L)\s+(?P<timestamp>(\d+))\s*'
BLINK_STOP_REGEX = (
    r'EBLINK\s+(R|L)\s+(?P<timestamp_start>(\d+))\s+'
    r'(?P<timestamp_end>(\d{6,10}))\s+(?P<duration_ms>(\d+))\s*'
)
INVALID_SAMPLE_REGEX = r'(?P<timestamp>(\d{6,10}))\s+\.\s+\.\s+0\.0\s+0\.0\s+\.\.\.\s*'

CALIBRATION_TIMESTAMP_REGEX = r'MSG\s+(?P<timestamp>\d+)\s+!CAL\s*\n'

CALIBRATION_REGEX = (
    r'>+\s+CALIBRATION\s+\(HV(?P<num_points>\d\d?),'
    r'(?P<type>.*)\).*'
    r'(?P<tracked_eye>RIGHT|LEFT):\s+<{9}'
)

START_RECORDING_REGEX = r'START\s+(?P<timestamp>(\d+))\s+(RIGHT|LEFT)\s+(?P<types>.*)'
STOP_RECORDING_REGEX = (
    r'END\s+(?P<timestamp>(\d+))\s+\s+(?P<types>.*)\s+RES\s+'
    r'(?P<xres>[\d\.]*)\s+(?P<yres>[\d\.]*)\s*'
)


def check_nan(sample_location: str) -> float:
    """Return position as float or np.nan depending on validity of sample.

    Parameters
    ----------
    sample_location: str
        Sample location as extracted from ascii file.

    Returns
    -------
    float
        Returns either the valid sample as a float or np.nan.
    """
    try:
        ret = float(sample_location)
    except ValueError:
        ret = np.nan
    return ret


def compile_patterns(patterns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compile patterns from strings.

    Parameters
    ----------
    patterns: list[dict[str, Any]]
        The list of patterns to compile.

    Returns
    -------
    list[dict[str, Any]]
        Returns from string compiled regex patterns.
    """
    msg_prefix = r'MSG\s+\d+\s+'

    compiled_patterns = []

    for pattern in patterns:
        if isinstance(pattern, str):
            compiled_pattern = {'pattern': re.compile(msg_prefix + pattern)}
            compiled_patterns.append(compiled_pattern)
            continue

        if isinstance(pattern, dict):
            if isinstance(pattern['pattern'], str):
                compiled_patterns.append({
                    **pattern,
                    'pattern': re.compile(msg_prefix + pattern['pattern']),
                })
                continue

            if isinstance(pattern['pattern'], tuple):
                for single_pattern in pattern['pattern']:
                    compiled_patterns.append({
                        **pattern,
                        'pattern': re.compile(msg_prefix + single_pattern),
                    })
                continue

            raise ValueError(f'invalid pattern: {pattern}')

        raise ValueError(f'invalid pattern: {pattern}')

    return compiled_patterns


def get_additional_columns(compiled_patterns: list[dict[str, Any]]) -> set[str]:
    """Get additionally needed columns from compiled patterns."""
    additional_columns = set()

    for compiled_pattern_dict in compiled_patterns:
        if 'column' in compiled_pattern_dict:
            additional_columns.add(compiled_pattern_dict['column'])

        for column in compiled_pattern_dict['pattern'].groupindex.keys():
            additional_columns.add(column)

    return additional_columns


def parse_eyelink(
        filepath: Path | str,
        patterns: list[dict[str, Any]] | None = None,
        schema: dict[str, Any] | None = None,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Process EyeLink asc file.

    Parameters
    ----------
    filepath: Path | str
        file name of ascii file to convert.
    patterns: list[dict[str, Any]] | None
        list of patterns to match for additional columns. (default: None)
    schema: dict[str, Any] | None
        Dictionary to optionally specify types of columns parsed by patterns. (default: None)

    Returns
    -------
    tuple[pl.DataFrame, dict[str, Any]]
        A tuple containing the parsed sample data and the metadata in a dictionary.

    Raises
    ------
    Warning
        If no metadata is found in the file.
    """
    if patterns is None:
        patterns = []
    compiled_patterns = compile_patterns(patterns)

    additional_columns = get_additional_columns(compiled_patterns)
    additional: dict[str, list[Any]] = {
        additional_column: [] for additional_column in additional_columns
    }
    current_additional = {
        additional_column: None for additional_column in additional_columns
    }

    samples: dict[str, list[Any]] = {
        'time': [],
        'x_pix': [],
        'y_pix': [],
        'pupil': [],
        **additional,
    }

    with open(filepath, encoding='ascii') as asc_file:
        lines = asc_file.readlines()

    # will return an empty string if the key does not exist
    metadata: defaultdict = defaultdict(str)

    compiled_metadata_patterns = []
    for metadata_pattern in EYELINK_META_REGEXES:
        compiled_metadata_patterns.append({'pattern': re.compile(metadata_pattern['pattern'])})

    compiled_validation_pattern = re.compile(VALIDATION_REGEX)
    compiled_calibration_pattern = re.compile(CALIBRATION_REGEX)
    compiled_calibration_timestamp = re.compile(CALIBRATION_TIMESTAMP_REGEX)
    cal_timestamp = ''

    compiled_blink_start = re.compile(BLINK_START_REGEX)
    compiled_blink_stop = re.compile(BLINK_STOP_REGEX)
    compiled_invalid_sample = re.compile(INVALID_SAMPLE_REGEX)

    compiled_recording_start = re.compile(START_RECORDING_REGEX)
    compiled_recording_stop = re.compile(STOP_RECORDING_REGEX)

    validations = []
    calibrations = []
    blinks = []
    invalid_samples = []

    blink = False

    start_recording_timestamp = ''
    total_recording_duration = 0
    num_blink_samples = 0

    for line in lines:

        for pattern_dict in compiled_patterns:

            if match := pattern_dict['pattern'].match(line):
                if 'value' in pattern_dict:
                    current_column = pattern_dict['column']
                    current_additional[current_column] = pattern_dict['value']

                else:
                    for column, value in match.groupdict().items():
                        current_additional[column] = value

        if cal_timestamp:
            # if a calibration timestamp has been found, the next line will be a
            # calibration pattern, if not, there will only be the timestamp added to the overview

            # very ugly pylint solution
            calibrations.append(
                {
                    'timestamp': cal_timestamp,
                    **match.groupdict(),
                }
                if (match := compiled_calibration_pattern.match(line))
                else {'timestamp': cal_timestamp},
            )
            cal_timestamp = ''

        elif compiled_blink_start.match(line):
            blink = True

        elif match := compiled_blink_stop.match(line):
            blink = False
            parsed_blink = match.groupdict()
            blink_info = {
                'start_timestamp': parsed_blink['timestamp_start'],
                'stop_timestamp': parsed_blink['timestamp_end'],
                'duration_ms': parsed_blink['duration_ms'],
                'num_samples': num_blink_samples,
            }
            num_blink_samples = 0
            blinks.append(blink_info)

        elif match := compiled_recording_start.match(line):
            start_recording_timestamp = match.groupdict()['timestamp']

        elif match := compiled_recording_stop.match(line):
            stop_recording_timestamp = match.groupdict()['timestamp']
            block_duration = int(stop_recording_timestamp) - int(start_recording_timestamp)

            total_recording_duration += block_duration

        elif eye_tracking_sample_match := EYE_TRACKING_SAMPLE.match(line):

            timestamp_s = eye_tracking_sample_match.group('time')
            x_pix_s = eye_tracking_sample_match.group('x_pix')
            y_pix_s = eye_tracking_sample_match.group('y_pix')
            pupil_s = eye_tracking_sample_match.group('pupil')

            timestamp = int(timestamp_s)
            x_pix = check_nan(x_pix_s)
            y_pix = check_nan(y_pix_s)
            pupil = check_nan(pupil_s)

            samples['time'].append(timestamp)
            samples['x_pix'].append(x_pix)
            samples['y_pix'].append(y_pix)
            samples['pupil'].append(pupil)

            for additional_column in additional_columns:
                samples[additional_column].append(current_additional[additional_column])

            if (match := compiled_invalid_sample.match(line)) and not blink:
                invalid_samples.append(match.groupdict()['timestamp'])
            elif compiled_invalid_sample.match(line) and blink:
                num_blink_samples += 1

        elif match := compiled_calibration_timestamp.match(line):
            cal_timestamp = match.groupdict()['timestamp']

        elif match := compiled_validation_pattern.match(line):
            validations.append(match.groupdict())

        elif compiled_metadata_patterns:
            for pattern_dict in compiled_metadata_patterns.copy():
                if match := pattern_dict['pattern'].match(line):
                    for column, value in match.groupdict().items():
                        metadata[column] = value

                    # each metadata pattern should only match once
                    compiled_metadata_patterns.remove(pattern_dict)

    if metadata:
        # if the sampling rate is not found, we cannot calculate the data loss
        actual_number_of_samples = len(samples['time'])

        data_loss_ratio, data_loss_ratio_blinks = _calculate_data_loss(
            blinks=blinks,
            invalid_samples=invalid_samples,
            actual_num_samples=actual_number_of_samples,
            total_rec_duration=total_recording_duration,
            sampling_rate=metadata['sampling_rate'],
        )

        pre_processed_metadata: dict[str, Any] = _pre_process_metadata(metadata)

        # is not yet pre-processed but should be
        pre_processed_metadata['calibrations'] = calibrations
        pre_processed_metadata['validations'] = validations
        pre_processed_metadata['blinks'] = blinks
        pre_processed_metadata['data_loss_ratio'] = data_loss_ratio
        pre_processed_metadata['data_loss_ratio_blinks'] = data_loss_ratio_blinks
        pre_processed_metadata['total_recording_duration_ms'] = total_recording_duration

    else:
        raise Warning('No metadata found. Please check the file for errors.')

    schema_overrides = {
        'time': pl.Int64,
        'x_pix': pl.Float64,
        'y_pix': pl.Float64,
        'pupil': pl.Float64,
    }
    if schema is not None:
        for column, dtype in schema.items():
            schema_overrides[column] = dtype

    df = pl.from_dict(
        data=samples,
        schema_overrides=schema_overrides,
    )

    return df, pre_processed_metadata


def _pre_process_metadata(metadata: defaultdict[str, Any]) -> dict[str, Any]:
    """Pre-process metadata to suitable types and formats.

    Parameters
    ----------
    metadata: defaultdict[str, Any]
        Metadata to pre-process.

    Returns
    -------
    dict[str, Any]
        Pre-processed metadata.
    """
    # in case the version strings have not been found, they will be empty strings (defaultdict)
    metadata['version_number'], metadata['model'] = _parse_full_eyelink_version(
        metadata['version_1'], metadata['version_2'],
    )

    if 'resolution' in metadata:
        coordinates = [int(coord) for coord in metadata['resolution'].split()]
        resolution = (coordinates[2] - coordinates[0] + 1, coordinates[3] - coordinates[1] + 1)
        metadata['resolution'] = resolution

    if metadata['sampling_rate']:
        metadata['sampling_rate'] = float(metadata['sampling_rate'])
    else:
        metadata['sampling_rate'] = 'unknown'

    # if the data has been parsed fully, convert the date to a datetime object
    if 'day' in metadata and 'year' in metadata and 'month' in metadata and 'time' in metadata:
        metadata['day'] = int(metadata['day'])
        metadata['year'] = int(metadata['year'])
        month_num = list(calendar.month_abbr).index(metadata['month'])
        date_time = datetime.datetime(day=metadata['day'], month=month_num, year=metadata['year'])
        time = datetime.datetime.strptime(metadata['time'], '%H:%M:%S')
        metadata['datetime'] = datetime.datetime.combine(date_time, time.time())

    return_metadata: dict[str, Any] = dict(metadata)

    return return_metadata


def _calculate_data_loss(
        blinks: list[dict[str, Any]],
        invalid_samples: list[str],
        actual_num_samples: int,
        total_rec_duration: int,
        sampling_rate: float,
) -> tuple[float | str, float | str]:
    """Calculate data loss and blink loss.

    Parameters
    ----------
    blinks: list[dict[str, Any]]
        List of dicts of blinks. Each dict containing start and stop timestamps and duration.
    invalid_samples: list[str]
        List of invalid samples.
    actual_num_samples: int
        Number of actual samples recorded.
    total_rec_duration: int
        Total duration of the recording.
    sampling_rate: float
        Sampling rate of the eye tracker.

    Returns
    -------
    tuple[float | str, float | str]
        Data loss ratio and blink loss ratio.
    """
    if not sampling_rate or not total_rec_duration:
        return 'unknown', 'unknown'

    dl_ratio_blinks = 0.0

    num_expected_samples = total_rec_duration * float(sampling_rate) / 1000

    total_lost_samples = num_expected_samples - actual_num_samples

    if blinks:
        total_blink_samples = sum(blink['num_samples'] for blink in blinks)
        dl_ratio_blinks = total_blink_samples / num_expected_samples
        total_lost_samples += total_blink_samples

    total_lost_samples += len(invalid_samples)

    dl_ratio = total_lost_samples / num_expected_samples

    return dl_ratio, dl_ratio_blinks


def _parse_full_eyelink_version(version_str_1: str, version_str_2: str) -> tuple[str, str]:
    """Parse the two version strings into an eyelink version number and model.

    Parameters
    ----------
    version_str_1: str
        First version string.
    version_str_2: str
        Second version string.

    Returns
    -------
    tuple[str, str]
        Version number and model as strings or unknown if it cannot be parsed.
    """
    if version_str_1 == 'EYELINK II 1' and version_str_2:
        version_pattern = re.compile(r'.*v(?P<version_number>[0-9]\.[0-9]+).*')
        if match := version_pattern.match(version_str_2):
            version_number = match.groupdict()['version_number']
            if float(version_number) < 3:
                model = 'EyeLink II'
            elif float(version_number) < 5:
                model = 'EyeLink 1000'
            elif float(version_number) < 6:
                model = 'EyeLink 1000 Plus'
            else:
                model = 'EyeLink Portable Duo'

        else:
            version_number = 'unknown'
            model = 'unknown'

    else:
        # taken from R package eyelinker/eyelink_parser.R
        version_pattern = re.compile(r'.*\s+(?P<version_number>[0-9]\.[0-9]+).*')
        model = 'EyeLink I'
        if match := version_pattern.match(version_str_1):
            version_number = match.groupdict()['version_number']

        else:
            model = 'unknown'
            version_number = 'unknown'

    return version_number, model
