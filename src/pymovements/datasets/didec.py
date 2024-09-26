# Copyright (c) 2022-2024 The pymovements Project Authors
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
"""Provides a definition for the DIDEC dataset."""
from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import Any

from pymovements.dataset.dataset_definition import DatasetDefinition
from pymovements.dataset.dataset_library import register_dataset
from pymovements.gaze.experiment import Experiment


@dataclass
@register_dataset
class DIDEC(DatasetDefinition):
    """DIDEC dataset :cite:p:`DIDEC`.

    The DIDEC eye-tracking data has two different data collections, (1) for the
    description viewing task is more coherent than for the free-viewing task;
    (2) variation in image descriptions. The data was collected using BeGaze eye-tracker
    with a sampling rate of 250 Hz. The data collection contains 112 Dutch students,
    54 students completed the free viewing task, while 58 completed the image description task.

    Check the respective paper for details :cite:p:`DIDEC`.

    Attributes
    ----------
    name: str
        The name of the dataset.

    mirrors: dict[str, tuple[str, ...]]
        A tuple of mirrors of the dataset. Each entry must be of type `str` and end with a '/'.

    resources: dict[str, tuple[dict[str, str], ...]]
        A tuple of dataset resources. Each list entry must be a dictionary with the following keys:
        - `resource`: The url suffix of the resource. This will be concatenated with the mirror.
        - `filename`: The filename under which the file is saved as.
        - `md5`: The MD5 checksum of the respective file.

    experiment: Experiment
        The experiment definition.

    filename_format: dict[str, str]
        Regular expression which will be matched before trying to load the file. Namedgroups will
        appear in the `fileinfo` dataframe.

    filename_format_dtypes: dict[str, dict[str, type]]
        If named groups are present in the `filename_format`, this makes it possible to cast
        specific named groups to a particular datatype.

    column_map: dict[str, str]
        The keys are the columns to read, the values are the names to which they should be renamed.

    custom_read_kwargs: dict[str, dict[str, Any]]
        If specified, these keyword arguments will be passed to the file reading function.

    Examples
    --------
    Initialize your :py:class:`~pymovements.PublicDataset` object with the
    :py:class:`~pymovements.DIDEC` definition:

    >>> import pymovements as pm
    >>>
    >>> dataset = pm.Dataset("DIDEC", path='data/DIDEC')

    Download the dataset resources:

    >>> dataset.download()# doctest: +SKIP

    Load the data into memory:

    >>> dataset.load()# doctest: +SKIP
    """

    # pylint: disable=similarities
    # The PublicDatasetDefinition child classes potentially share code chunks for definitions.

    name: str = 'DIDEC'

    has_files: dict[str, bool] = field(
        default_factory=lambda: {
            'gaze': True,
            'precomputed_events': False,
            'precomputed_reading_measures': False,
        },
    )

    mirrors: dict[str, tuple[str, ...]] = field(
        default_factory=lambda: {
            'gaze': ('https://didec.uvt.nl/corpus/',),
        },
    )

    resources: dict[str, tuple[dict[str, str], ...]] = field(
        default_factory=lambda: {
            'gaze': (
                {
                    'resource': 'DIDEC_only_the_eyetracking_data.zip',
                    'filename': 'DIDEC_only_the_eyetracking_data.zip',
                    'md5': 'd572b0b41828986ca48a2fcf6966728a',
                },
            ),
        },
    )

    extract: dict[str, bool] = field(default_factory=lambda: {'gaze': True})

    experiment: Experiment = Experiment(
        screen_width_px=1680,
        screen_height_px=1050,
        screen_width_cm=47.4,
        screen_height_cm=29.7,
        distance_cm=70,
        origin='upper left',
        sampling_rate=1000,
    )

    filename_format: dict[str, str] = field(
        default_factory=lambda: {
            'gaze':
                (
                    r'Ruud_exp{experiment:d}_'
                    r'list{list:d}_v{version:d}_'
                    r'ppn{participant:d}_{session:d}_'
                    r'Trial{trial:d} Samples.txt'
                ),
        },
    )

    filename_format_dtypes: dict[str, dict[str, type]] = field(
        default_factory=lambda: {
            'gaze': {
                'experiment': int,
                'list': int,
                'version': int,
                'participant': int,
                'session': int,
                'trial': int,
            },
        },
    )

    trial_columns: list[str] = field(
        default_factory=lambda: ['Stimulus'],
    )

    time_column: str = 'Time'

    time_unit: str = 'ms'

    pixel_columns: list[str] = field(
        default_factory=lambda: [
            'L POR X [px]',
            'L POR Y [px]',
            'R POR X [px]',
            'R POR Y [px]',
        ],
    )

    column_map: dict[str, str] = field(default_factory=lambda: {})

    custom_read_kwargs: dict[str, dict[str, Any]] = field(
        default_factory=lambda: {
            'gaze': {
                'separator': '\t',
                # skip begaze tracker data
                'skip_rows': 43,
                'has_header': False,
                'new_columns': [
                    'Time',
                    'Type',
                    'Trial',
                    'L POR X [px]',
                    'L POR Y [px]',
                    'R POR X [px]',
                    'R POR Y [px]',
                    'Timing',
                    'Pupil Confidence',
                    'L Plane',
                    'R Plane',
                    'L Event Info',
                    'R Event Info',
                    'Stimulus',
                ],
            },
        },
    )
