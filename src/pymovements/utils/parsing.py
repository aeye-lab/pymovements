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
from typing import Any

import numpy as np
import polars as pl

EYE_TRACKING_SAMPLE = re.compile(
    r'(?P<time>(\d{7,10}))\s+'
    r'(?P<x_pix>[-]?\d*[.]\d*)\s+'
    r'(?P<y_pix>[-]?\d*[.]\d*)\s+'
    r'(?P<pupil>\d*[.]\d*)\s+'
    r'(?P<dummy>\d*[.]\d*)\s+'
    r'(?P<dots>[A-Za-z.]{3,5})?\s*',
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


def compile_patterns(patterns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compile patterns from strings.

    Parameters
    ----------
    patterns:
        The list of patterns to compile.
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
        filepath: Path,
        patterns: list | None = None,
        schema: dict | None = None,
) -> pl.DataFrame:
    """Processes ascii files to csv.

    Parameters
    ----------
    filepath:
        file name of ascii file to convert.
    patterns:
        list of patterns to match for additional columns.
    schema:
        Dictionary to optionally specify types of columns parsed by patterns.

    """
    if patterns is None:
        patterns = []
    compiled_patterns = compile_patterns(patterns)

    additional_columns = get_additional_columns(compiled_patterns)
    additional: dict[str, list] = {
        additional_column: [] for additional_column in additional_columns
    }
    current_additional = {
        additional_column: None for additional_column in additional_columns
    }

    samples: dict[str, list] = {
        'time': [],
        'x_pix': [],
        'y_pix': [],
        'pupil': [],
        **additional,
    }

    with open(filepath, encoding='ascii') as asc_file:
        lines = asc_file.readlines()

    for line in lines:
        for pattern_dict in compiled_patterns:
            match = pattern_dict['pattern'].match(line)

            if match:
                if 'value' in pattern_dict:
                    current_column = pattern_dict['column']
                    current_additional[current_column] = pattern_dict['value']

                else:
                    for column, value in match.groupdict().items():
                        current_additional[column] = value

        eye_tracking_sample_match = EYE_TRACKING_SAMPLE.match(line)
        if eye_tracking_sample_match:

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

    df = pl.from_dict(
        data=samples,
        schema_overrides={
            'time': pl.Int64,
            'x_pix': pl.Float64,
            'y_pix': pl.Float64,
            'pupil': pl.Float64,
        },
    )

    if schema is not None:
        df = df.with_columns([pl.col(column).cast(dtype) for column, dtype in schema.items()])

    return df
