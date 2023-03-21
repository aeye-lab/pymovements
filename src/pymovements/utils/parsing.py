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

import os
import re
from pathlib import Path

import numpy as np
import polars as pl

EYE_TRACKING_SAMPLE = re.compile(
    r'(?P<time>(\d{7}|\d{8}|\d{9}|\d{10}))\s+'
    r'(?P<x_eye>[-]?\d*[.]\d*)\s+'
    r'(?P<y_eye>[-]?\d*[.]\d*)\s+'
    r'(?P<pupil_eye>\d*[.]\d*)\s+'
    r'(?P<dummy>\d*[.]\d*)\s+'
    r'((?P<dots>[A-Za-z.]{5}))?\s*',
)


class scolor:
    """Implements colors for conversion"""
    created: str = '\x1b[1;37;42m'
    skipped: str = '\x1b[1;37;44m'
    rewritten: str = '\x1b[1;37;43m'
    end: str = '\x1b[0m'


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


def get_eye_tracking_sample_feature(eye_tracking_sample_match: re.Match, feature: str) -> str:
    """Extract sample information from line.

    Parameters
    ----------
    eye_tracking_sample_match: re.Match
        Matched eye tracking sample pattern.
    feature: str
        feature to extract from matched eye tracking sample pattern.
    """
    return eye_tracking_sample_match.group(feature)


def process_asc2csv(
    file_name: Path,
    sync_msg_start_pattern: None | str = None,
    sync_msg_stop_pattern: None | str = None,
    overwrite_existing: bool = False,
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
    overwrite_existing: bool,
        Decide whether to overwrite an existing csv file.
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
    csv_file_name = f"{Path(*file_name.parts[:-1])}/{file_name.parts[-1].split('.')[0]}.csv"
    df = pl.DataFrame(
        schema=[
            ('time', pl.Int64),
            ('x_eye', pl.Float64),
            ('y_eye', pl.Float64),
            ('pupil_eye', pl.Float64),
            ('task', pl.Utf8),
        ],
    )
    if not os.path.exists(csv_file_name):
        status_msg = 'Created'
        status_clr = scolor.created
    elif os.path.exists(csv_file_name) and not overwrite_existing:
        status_msg = 'Skipped'
        status_clr = scolor.skipped
        written_msg = f'Processing csv for {file_name}'.ljust(105 - len(status_msg), '.')
        print(written_msg, end='', flush=True)
        print(f'{status_clr}{status_msg}{scolor.end}', flush=True)
        return df
    else:
        status_msg = 'Rewritten'
        status_clr = scolor.rewritten

    written_msg = f'Processing csv for {file_name}'.ljust(105 - len(status_msg), '.')
    print(written_msg, end='', flush=True)
    with open(file_name, encoding='ascii') as asc_file:
        lines = asc_file.readlines()
    cur_task_type = None
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
                cur_task_type = str(sync_msg_start_pattern) + start_id
            else:
                cur_task_type = f'TRIALID_{start_id}'
            during_task_recording = True
        if during_task_recording:
            if EYE_TRACKING_SAMPLE.match(line):
                eye_tracking_sample_match = EYE_TRACKING_SAMPLE.match(line)
                # mypy is unaware that 'eye_tracking_sample_match' can never be None (l.156)
                assert eye_tracking_sample_match is not None

                timestamp_s = get_eye_tracking_sample_feature(
                    eye_tracking_sample_match,
                    'time',
                )
                x_eye_s = get_eye_tracking_sample_feature(
                    eye_tracking_sample_match,
                    'x_eye',
                )
                y_eye_s = get_eye_tracking_sample_feature(
                    eye_tracking_sample_match,
                    'y_eye',
                )
                pupil_eye_s = get_eye_tracking_sample_feature(
                    eye_tracking_sample_match,
                    'pupil_eye',
                )

                timestamp = int(timestamp_s)
                x_eye = check_nan(x_eye_s)
                y_eye = check_nan(y_eye_s)
                pupil_eye = check_nan(pupil_eye_s)
                df = df.extend(
                    pl.DataFrame(
                        {
                            'time': [timestamp],
                            'x_eye': [x_eye],
                            'y_eye': [y_eye],
                            'pupil_eye': [pupil_eye],
                            'task': [cur_task_type],
                        },
                    ),
                )

    df.write_csv(csv_file_name, separator='\t')
    print(f'{status_clr}{status_msg}{scolor.end}', flush=True)
    return df
