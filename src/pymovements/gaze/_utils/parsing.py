# Copyright (c) 2023-2025 The pymovements Project Authors
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
    r'(?P<time>(\d+[.]?\d*))\s+'
    r'(?P<x_pix>[-]?\d*[.]\d*)\s+'
    r'(?P<y_pix>[-]?\d*[.]\d*)\s+'
    r'(?P<pupil>\d*[.]\d*)\s+'
    r'((?P<dummy>\d*[.]\d*)\s+)?'  # optional dummy column
    r'(?P<dots>[A-Za-z.]{3,5})?\s*',
)

EYELINK_META_REGEXES = [
    {'pattern': re.compile(regex)} for regex in (
        r'\*\*\s+VERSION:\s+(?P<version_1>.*)\s+',
        (
            r'\*\*\s+DATE:\s+(?P<weekday>[A-Z,a-z]+)\s+(?P<month>[A-Z,a-z]+)'
            r'\s+(?P<day>\d\d?)\s+(?P<time>\d\d:\d\d:\d\d)\s+(?P<year>\d{4})\s*'
        ),
        r'\*\*\s+(?P<version_2>EYELINK.*)',
        r'MSG\s+\d+[.]?\d*\s+DISPLAY_COORDS\s*=?\s*(?P<resolution>.*)',
        (
            r'MSG\s+\d+[.]?\d*\s+RECCFG\s+(?P<tracking_mode>[A-Z,a-z]+)\s+'
            r'(?P<sampling_rate>\d+)\s+'
            r'(?P<file_sample_filter>(0|1|2))\s+'
            r'(?P<link_sample_filter>(0|1|2))\s+'
            r'(?P<tracked_eye>(L|R|LR))\s*'
        ),
        r'PUPIL\s+(?P<pupil_data_type>(AREA|DIAMETER))\s*',
        r'MSG\s+\d+[.]?\d*\s+ELCLCFG\s+(?P<mount_configuration>.*)',
    )
]

VALIDATION_REGEX = re.compile(
    r'MSG\s+(?P<timestamp>\d+[.]?\d*)\s+!CAL\s+VALIDATION\s+HV'
    r'(?P<num_points>\d\d?).*'
    r'(?P<tracked_eye>LEFT|RIGHT)\s+'
    r'(?P<error>\D*)\s+'
    r'(?P<validation_score_avg>\d.\d\d)\s+avg\.\s+'
    r'(?P<validation_score_max>\d.\d\d)\s+max',
)

BLINK_START_REGEX = re.compile(r'SBLINK\s+(R|L)\s+(?P<timestamp>(\d+[.]?\d*))\s*')
BLINK_STOP_REGEX = re.compile(
    r'EBLINK\s+(R|L)\s+(?P<timestamp_start>(\d+[.]?\d*))\s+'
    r'(?P<timestamp_end>(\d+[.]?\d*))\s+(?P<duration_ms>(\d+[.]?\d*))\s*',
)
INVALID_SAMPLE_REGEX = re.compile(
    r'(?P<timestamp>(\d+[.]?\d*))\s+\.\s+\.\s+(?P<dummy>0\.0)?\s+0\.0\s+\.\.\.\s*',
)

CALIBRATION_TIMESTAMP_REGEX = re.compile(r'MSG\s+(?P<timestamp>\d+[.]?\d*)\s+!CAL\s*\n')

CALIBRATION_REGEX = re.compile(
    r'>+\s+CALIBRATION\s+\(HV(?P<num_points>\d\d?),'
    r'(?P<type>.*)\).*'
    r'(?P<tracked_eye>RIGHT|LEFT):\s+<{9}',
)

START_RECORDING_REGEX = re.compile(
    r'START\s+(?P<timestamp>(\d+[.]?\d*))\s+(RIGHT|LEFT)\s+(?P<types>.*)',
)
STOP_RECORDING_REGEX = re.compile(
    r'END\s+(?P<timestamp>(\d+[.]?\d*))\s+\s+(?P<types>.*)\s+RES\s+'
    r'(?P<xres>[\d\.]*)\s+(?P<yres>[\d\.]*)\s*',
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


def compile_patterns(patterns: list[dict[str, Any] | str]) -> list[dict[str, Any]]:
    """Compile patterns from strings.

    Parameters
    ----------
    patterns: list[dict[str, Any] | str]
        The list of patterns to compile.

    Returns
    -------
    list[dict[str, Any]]
        Returns from string compiled regex patterns.
    """
    msg_prefix = r'MSG\s+\d+[.]?\d*\s+'

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

            if isinstance(pattern['pattern'], (tuple, list)):
                for single_pattern in pattern['pattern']:
                    compiled_patterns.append({
                        **pattern,
                        'pattern': re.compile(msg_prefix + single_pattern),
                    })
                continue

            raise ValueError(f'invalid pattern: {pattern}')

        raise ValueError(f'invalid pattern: {pattern}')

    return compiled_patterns


def get_pattern_keys(compiled_patterns: list[dict[str, Any]], pattern_key: str) -> set[str]:
    """Get names of capture groups or column/metadata keys."""
    keys = set()

    for compiled_pattern_dict in compiled_patterns:
        if pattern_key in compiled_pattern_dict:
            keys.add(compiled_pattern_dict[pattern_key])

        for key in compiled_pattern_dict['pattern'].groupindex.keys():
            keys.add(key)

    return keys


def parse_eyelink(
        filepath: Path | str,
        patterns: list[dict[str, Any] | str] | None = None,
        schema: dict[str, Any] | None = None,
        metadata_patterns: list[dict[str, Any] | str] | None = None,
        encoding: str = 'ascii',
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Parse EyeLink asc file.

    Parameters
    ----------
    filepath: Path | str
        file name of ascii file to convert.
    patterns: list[dict[str, Any] | str] | None
        List of patterns to match for additional columns. (default: None)
    schema: dict[str, Any] | None
        Dictionary to optionally specify types of columns parsed by patterns. (default: None)
    metadata_patterns: list[dict[str, Any] | str] | None
        list of patterns to match for additional metadata. (default: None)
    encoding: str
        Text encoding of the file. (default: 'ascii')

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

    if metadata_patterns is None:
        metadata_patterns = []
    compiled_metadata_patterns = compile_patterns(metadata_patterns)

    additional_columns = get_pattern_keys(compiled_patterns, 'column')
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

    with open(filepath, encoding=encoding) as asc_file:
        lines = asc_file.readlines()

    # will return an empty string if the key does not exist
    metadata: defaultdict = defaultdict(str)

    # metadata keys specified by the user should have a default value of None
    metadata_keys = get_pattern_keys(compiled_metadata_patterns, 'key')
    for key in metadata_keys:
        metadata[key] = None

    compiled_metadata_patterns.extend(EYELINK_META_REGEXES)

    cal_timestamp = ''

    validations = []
    calibrations = []
    blinks = []
    invalid_samples = []

    blink = False

    start_recording_timestamp = ''
    total_recording_duration = 0.0
    num_blink_samples = 0

    for line in lines:

        for pattern_dict in compiled_patterns:

            if match := pattern_dict['pattern'].match(line):
                if 'value' in pattern_dict:
                    current_column = pattern_dict['column']
                    current_additional[current_column] = pattern_dict['value']

                else:
                    current_additional.update(match.groupdict())

        if cal_timestamp:
            # if a calibration timestamp has been found, the next line will be a
            # calibration pattern, if not, there will only be the timestamp added to the overview

            # very ugly pylint solution
            calibrations.append(
                {
                    'timestamp': cal_timestamp,
                    **match.groupdict(),
                }
                if (match := CALIBRATION_REGEX.match(line))
                else {'timestamp': cal_timestamp},
            )
            cal_timestamp = ''

        elif BLINK_START_REGEX.match(line):
            blink = True

        elif match := BLINK_STOP_REGEX.match(line):
            blink = False
            parsed_blink = match.groupdict()
            blink_info = {
                'start_timestamp': float(parsed_blink['timestamp_start']),
                'stop_timestamp': float(parsed_blink['timestamp_end']),
                'duration_ms': float(parsed_blink['duration_ms']),
                'num_samples': num_blink_samples,
            }
            num_blink_samples = 0
            blinks.append(blink_info)

        elif match := START_RECORDING_REGEX.match(line):
            start_recording_timestamp = match.groupdict()['timestamp']

        elif match := STOP_RECORDING_REGEX.match(line):
            stop_recording_timestamp = match.groupdict()['timestamp']
            block_duration = float(stop_recording_timestamp) - float(start_recording_timestamp)

            total_recording_duration += block_duration

        elif eye_tracking_sample_match := EYE_TRACKING_SAMPLE.match(line):

            timestamp_s = eye_tracking_sample_match.group('time')
            x_pix_s = eye_tracking_sample_match.group('x_pix')
            y_pix_s = eye_tracking_sample_match.group('y_pix')
            pupil_s = eye_tracking_sample_match.group('pupil')

            timestamp = float(timestamp_s)
            x_pix = check_nan(x_pix_s)
            y_pix = check_nan(y_pix_s)
            pupil = check_nan(pupil_s)

            samples['time'].append(timestamp)
            samples['x_pix'].append(x_pix)
            samples['y_pix'].append(y_pix)
            samples['pupil'].append(pupil)

            for additional_column in additional_columns:
                samples[additional_column].append(current_additional[additional_column])

            if match := INVALID_SAMPLE_REGEX.match(line):
                if blink:
                    num_blink_samples += 1
                else:
                    invalid_samples.append(match.groupdict()['timestamp'])

        elif match := CALIBRATION_TIMESTAMP_REGEX.match(line):
            cal_timestamp = match.groupdict()['timestamp']

        elif match := VALIDATION_REGEX.match(line):
            validations.append(match.groupdict())

        elif compiled_metadata_patterns:
            for pattern_dict in compiled_metadata_patterns.copy():
                if match := pattern_dict['pattern'].match(line):
                    if 'value' in pattern_dict:
                        metadata[pattern_dict['key']] = pattern_dict['value']

                    else:
                        metadata.update(match.groupdict())

                    # each metadata pattern should only match once
                    compiled_metadata_patterns.remove(pattern_dict)

    if not metadata:
        raise Warning('No metadata found. Please check the file for errors.')

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

    schema_overrides = {
        'time': pl.Float64,
        'x_pix': pl.Float64,
        'y_pix': pl.Float64,
        'pupil': pl.Float64,
    }
    if schema is not None:
        schema_overrides.update(schema)

    df = pl.from_dict(data=samples).cast(schema_overrides)

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

    # if the date has been parsed fully, convert the date to a datetime object
    if 'day' in metadata and 'year' in metadata and 'month' in metadata and 'time' in metadata:
        metadata['day'] = int(metadata['day'])
        metadata['year'] = int(metadata['year'])
        month_num = list(calendar.month_abbr).index(metadata['month'])
        date_time = datetime.datetime(day=metadata['day'], month=month_num, year=metadata['year'])
        time = datetime.datetime.strptime(metadata['time'], '%H:%M:%S')
        metadata['datetime'] = datetime.datetime.combine(date_time, time.time())

    if 'mount_configuration' in metadata:
        metadata['mount_configuration'] = _parse_eyelink_mount_config(
            metadata['mount_configuration'],
        )

    return_metadata: dict[str, Any] = dict(metadata)

    return return_metadata


def _calculate_data_loss(
        blinks: list[dict[str, Any]],
        invalid_samples: list[str],
        actual_num_samples: int,
        total_rec_duration: float,
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
    total_rec_duration: float
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


def _parse_eyelink_mount_config(mount_config: str) -> dict[str, str]:
    """Return a dictionary with the mount configuration based on the config short name.

    Parameters
    ----------
    mount_config: str
        Short name of the mount configuration.

    Returns
    -------
    dict[str, str]
        Dictionary with the mount configuration spelled out.

    """
    possible_mounts = {
        'MTABLER': {
            'mount_type': 'Desktop',
            'head_stabilization': 'stabilized',
            'eyes_recorded': 'monocular',
            'short_name': 'MTABLER',
        },
        'BTABLER': {
            'mount_type': 'Desktop',
            'head_stabilization': 'stabilized',
            'eyes_recorded': 'binocular / monocular',
            'short_name': 'BTABLER',
        },
        'RTABLER': {
            'mount_type': 'Desktop',
            'head_stabilization': 'remote',
            'eyes_recorded': 'monocular',
            'short_name': 'RTABLER',
        },
        'RBTABLER': {
            'mount_type': 'Desktop',
            'head_stabilization': 'remote',
            'eyes_recorded': 'binocular / monocular',
            'short_name': 'RBTABLER',
        },
        'AMTABLER': {
            'mount_type': 'Arm Mount',
            'head_stabilization': 'stabilized',
            'eyes_recorded': 'monocular',
            'short_name': 'AMTABLER',
        },
        'ABTABLER': {
            'mount_type': 'Arm Mount',
            'head_stabilization': 'stabilized',
            'eyes_recorded': 'binocular / monocular',
            'short_name': 'ABTABLER',
        },
        'ARTABLER': {
            'mount_type': 'Arm Mount',
            'head_stabilization': 'remote',
            'eyes_recorded': 'monocular',
            'short_name': 'ARTABLER',
        },
        'ABRTABLE': {
            'mount_type': 'Arm Mount',
            'head_stabilization': 'remote',
            'eyes_recorded': 'binocular / monocular',
            'short_name': 'ABRTABLE',
        },
        'BTOWER': {
            'mount_type': 'Binocular Tower Mount',
            'head_stabilization': 'stabilized',
            'eyes_recorded': 'binocular / monocular',
            'short_name': 'BTOWER',
        },
        'TOWER': {
            'mount_type': 'Tower Mount',
            'head_stabilization': 'stabilized',
            'eyes_recorded': 'monocular',
            'short_name': 'TOWER',
        },
        'MPRIM': {
            'mount_type': 'Primate Mount',
            'head_stabilization': 'stabilized',
            'eyes_recorded': 'monocular',
            'short_name': 'MPRIM',
        },
        'BPRIM': {
            'mount_type': 'Primate Mount',
            'head_stabilization': 'stabilized',
            'eyes_recorded': 'binocular / monocular',
            'short_name': 'BPRIM',
        },
        'MLRR': {
            'mount_type': 'Long-Range Mount',
            'head_stabilization': 'stabilized',
            'eyes_recorded': 'monocular',
            'camera_position': 'level',
            'short_name': 'MLRR',
        },
        'BLRR': {
            'mount_type': 'Long-Range Mount',
            'head_stabilization': 'stabilized',
            'eyes_recorded': 'binocular / monocular',
            'camera_position': 'angled',
            'short_name': 'BLRR',
        },
    }

    if mount_config in possible_mounts:
        return possible_mounts[mount_config]

    return {
        'mount_type': 'unknown',
        'head_stabilization': 'unknown',
        'eyes_recorded': 'unknown',
        'camera_position': 'unknown',
        'short_name': mount_config,
    }
