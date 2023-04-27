# Copyright (c) 2023 The pymovements Project Authors
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
""" Converts ascii to csv files."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import polars as pl

EYE_TRACKING_SAMPLE = re.compile(
    r'(?P<time>(\d{7}|\d{8}|\d{9}|\d{10}))\s+'
    r'(?P<x_pix>[-]?\d*[.]\d*)\s+'
    r'(?P<y_pix>[-]?\d*[.]\d*)\s+'
    r'(?P<pupil>\d*[.]\d*)\s+'
    r'(?P<dummy>\d*[.]\d*)\s+'
    r'(?P<dots>[A-Za-z.]{5})?\s*',
)


def check_nan(sample_location: str) -> float:
    """Returns position as float or np.nan depending on validity of sample

    Parameters
    ----------
    sample_location: str
        Sample location as extracted from ascii file.
    """

    try:
        ret = float(sample_location)
    except ValueError:
        ret = np.nan
    return ret


def parse_eyelink(
        file_name: Path,
        sync_msg_start_pattern: None | str = None,
        sync_msg_stop_pattern: None | str = None,
) -> pl.DataFrame:
    """Processes ascii files to csv.

    Parameters
    ----------
    file_name: Path
        file name of ascii file to convert
    sync_msg_start_pattern: str,
        Optional starting pattern of trial as sync message,
        if None is given reverts back to TRIALID
    sync_msg_stop_pattern: str,
        Optional stopping pattern of trial as sync message,
        if None is given reverts back to previous TRIALID
    """
    if sync_msg_start_pattern is None:
        # if no specific type pattern revert to trial ids
        _sync_msg_start_pattern = re.compile('MSG\t\\d+ TRIALID (?P<startpattern>[0-9]+)\n')
    else:
        _sync_msg_start_pattern = re.compile(
            f'MSG\t\\d+ .\\d+ {sync_msg_start_pattern}(?P<startpattern>[0-9]+)',
        )
    if sync_msg_stop_pattern is None:
        # if no stop pattern, stop when "next" occurance of start pattern
        _sync_msg_stop_pattern = _sync_msg_start_pattern
    else:
        _sync_msg_stop_pattern = re.compile(
            fr'MSG\t\d+\ \d+\ \S+(?P<stoppattern>{sync_msg_stop_pattern})\n',
        )

    samples: dict[str, list] = {
        'time': [],
        'x_pix': [],
        'y_pix': [],
        'pupil': [],
        'task': [],
    }

    with open(file_name, encoding='ascii') as asc_file:
        lines = asc_file.readlines()
    current_task = None
    during_task_recording = False
    for line in lines:

        if _sync_msg_stop_pattern.match(line):
            during_task_recording = False
        if _sync_msg_start_pattern.match(line):
            start_matched = _sync_msg_start_pattern.match(line)
            # mypy is unaware that 'start_matched' can never be None (l.146)
            assert start_matched is not None
            start_id = start_matched.group('startpattern')
            if sync_msg_start_pattern is not None:
                current_task = str(sync_msg_start_pattern) + start_id
            else:
                current_task = f'TRIALID_{start_id}'
            during_task_recording = True
        if during_task_recording:
            if EYE_TRACKING_SAMPLE.match(line):
                eye_tracking_sample_match = EYE_TRACKING_SAMPLE.match(line)
                # mypy is unaware that 'eye_tracking_sample_match' can never be None (l.156)
                assert eye_tracking_sample_match is not None

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
                samples['task'].append(current_task)

    df = pl.from_dict(
        data=samples,
        schema={
            'time': pl.Int64,
            'x_pix': pl.Float64,
            'y_pix': pl.Float64,
            'pupil': pl.Float64,
            'task': pl.Utf8,
        },
    )

    return df
