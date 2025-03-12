# Copyright (c) 2022-2025 The pymovements Project Authors
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
"""Provides a definition for the FakeNewsPerception dataset."""
from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import Any

from pymovements.dataset.dataset_definition import DatasetDefinition
from pymovements.gaze.experiment import Experiment


@dataclass
class FakeNewsPerception(DatasetDefinition):
    """FakeNewsPerception dataset :cite:p:`FakeNewsPerception`.

    FakeNewsPerception dataset consists of eye movements during reading,
    perceived believability scores, and questionnaires including Cognitive Reflection Test (CRT)
    and News-Find-Me (NFM) perception, collected from 25 participants with 60 news items.
    Eye movements are recorded to provide objective measures
    of information processing during news reading.

    For more details see :cite:p:`FakeNewsPerception`.

    Attributes
    ----------
    name: str
        The name of the dataset.
    has_files: dict[str, bool]
        Indicate whether the dataset contains 'gaze', 'precomputed_events', and
        'precomputed_reading_measures'.
    mirrors: dict[str, list[str]]
        A list of mirrors of the dataset. Each entry must be of type `str` and end with a '/'.
    resources: dict[str, list[dict[str, str]]]
        A list of dataset gaze_resources. Each list entry must be a dictionary with the following
        keys:
        - `resource`: The url suffix of the resource. This will be concatenated with the mirror.
        - `filename`: The filename under which the file is saved as.
        - `md5`: The MD5 checksum of the respective file.
    experiment: Experiment
        The experiment definition.
    extract: dict[str, bool]
        Decide whether to extract the data.
    filename_format: dict[str, str]
        Regular expression which will be matched before trying to load the file. Namedgroups will
        appear in the `fileinfo` dataframe.
    filename_format_schema_overrides: dict[str, dict[str, type]]
        If named groups are present in the `filename_format`, this makes it possible to cast
        specific named groups to a particular datatype.
    column_map: dict[str, str]
        The keys are the columns to read, the values are the names to which they should be renamed.
    custom_read_kwargs: dict[str, Any]
        If specified, these keyword arguments will be passed to the file reading function.
    """

    name: str = 'FakeNewsPerception'

    has_files: dict[str, bool] = field(
        default_factory=lambda: {
            'gaze': False,
            'precomputed_events': True,
            'precomputed_reading_measures': False,
        },
    )

    mirrors: dict[str, list[str]] = field(
        default_factory=lambda: {
            'precomputed_events': ['https://dataverse.harvard.edu/'],
        },
    )

    resources: dict[str, list[dict[str, str]]] = field(
        default_factory=lambda: {
            'precomputed_events': [
                {
                    'resource': 'api/access/datafile/4200164',
                    'filename': 'D3-Eye-movements-data.zip',
                    'md5': 'ab009f28cd703f433e9b6c02b0bb38d2',
                },
            ],
        },
    )

    experiment: Experiment = field(
        default_factory=lambda: Experiment(
            screen_width_px=1920,
            screen_height_px=1080,
            screen_width_cm=52.7,
            screen_height_cm=29.6,
            distance_cm=None,
            origin=None,
            sampling_rate=600,
        ),
    )

    extract: dict[str, bool] = field(default_factory=lambda: {'precomputed_events': True})

    filename_format: dict[str, str] = field(
        default_factory=lambda: {
            'precomputed_events': r'P{subject_id:d}_S{session_id:d}_{truth_value:s}.csv',
        },
    )

    filename_format_schema_overrides: dict[str, dict[str, type]] = field(
        default_factory=lambda: {
            'precomputed_events': {'subject_id': int, 'session_id': int, 'truth_value': str},
        },
    )

    column_map: dict[str, str] = field(default_factory=lambda: {})

    custom_read_kwargs: dict[str, Any] = field(
        default_factory=lambda: {
            'precomputed_events': {
                'null_values': 'NA',
                'quote_char': '"',
            },
        },
    )
