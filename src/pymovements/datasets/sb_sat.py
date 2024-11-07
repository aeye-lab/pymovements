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
"""Provides a definition for the SB-SAT dataset."""
from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import Any

import polars as pl

from pymovements.dataset.dataset_definition import DatasetDefinition
from pymovements.dataset.dataset_library import register_dataset
from pymovements.gaze.experiment import Experiment


@dataclass
@register_dataset
class SBSAT(DatasetDefinition):
    """SB-SAT dataset :cite:p:`SB-SAT`.

    This dataset includes monocular eye tracking data from a single participants in a single
    session. Eye movements are recorded at a sampling frequency of 1,000 Hz using an EyeLink 1000
    eye tracker and are provided as pixel coordinates.

    The participant is instructed to read texts and answer questions.

    Check the respective paper for details :cite:p:`SB-SAT`.

    Attributes
    ----------
    name: str
        The name of the dataset.

    has_files: dict[str, bool]
        Indicate whether the dataset contains 'gaze', 'precomputed_events', and
        'precomputed_reading_measures'.

    mirrors: dict[str, tuple[str, ...]]
        A tuple of mirrors of the dataset. Each entry must be of type `str` and end with a '/'.

    resources: dict[str, tuple[dict[str, str], ...]]
        A tuple of dataset gaze_resources. Each list entry must be a dictionary with the following
        keys:
        - `resource`: The url suffix of the resource. This will be concatenated with the mirror.
        - `filename`: The filename under which the file is saved as.
        - `md5`: The MD5 checksum of the respective file.

    extract: dict[str, bool]
        Decide whether to extract the data.

    experiment: Experiment
        The experiment definition.

    filename_format: dict[str, str]
        Regular expression which will be matched before trying to load the file. Namedgroups will
        appear in the `fileinfo` dataframe.

    filename_format_schema_overrides: dict[str, dict[str, type]]
        If named groups are present in the `filename_format`, this makes it possible to cast
        specific named groups to a particular datatype.

    trial_columns: list[str]
            The name of the trial columns in the input data frame. If the list is empty or None,
            the input data frame is assumed to contain only one trial. If the list is not empty,
            the input data frame is assumed to contain multiple trials and the transformation
            methods will be applied to each trial separately.

    time_column: str
        The name of the timestamp column in the input data frame. This column will be renamed to
        ``time``.

    time_unit: str
        The unit of the timestamps in the timestamp column in the input data frame. Supported
        units are 's' for seconds, 'ms' for milliseconds and 'step' for steps. If the unit is
        'step' the experiment definition must be specified. All timestamps will be converted to
        milliseconds.

    pixel_columns: list[str]
        The name of the pixel position columns in the input data frame. These columns will be
        nested into the column ``pixel``. If the list is empty or None, the nested ``pixel``
        column will not be created.

    column_map: dict[str, str]
        The keys are the columns to read, the values are the names to which they should be renamed.

    custom_read_kwargs: dict[str, dict[str, Any]]
        If specified, these keyword arguments will be passed to the file reading function.

    Examples
    --------
    Initialize your :py:class:`~pymovements.PublicDataset` object with the
    :py:class:`~pymovements.SBSAT` definition:

    >>> import pymovements as pm
    >>>
    >>> dataset = pm.Dataset("SBSAT", path='data/SBSAT')

    Download the dataset resources:

    >>> dataset.download()# doctest: +SKIP

    Load the data into memory:

    >>> dataset.load()# doctest: +SKIP
    """

    # pylint: disable=similarities
    # The PublicDatasetDefinition child classes potentially share code chunks for definitions.

    name: str = 'SBSAT'

    has_files: dict[str, bool] = field(
        default_factory=lambda: {
            'gaze': True,
            'precomputed_events': True,
            'precomputed_reading_measures': False,
        },
    )
    mirrors: dict[str, tuple[str, ...]] = field(
        default_factory=lambda:
            {
                'gaze': (
                    'https://osf.io/download/',
                ),
                'precomputed_events': (
                    'https://raw.githubusercontent.com/ahnchive/SB-SAT/master/fixation/',
                ),
            },
    )
    resources: dict[str, tuple[dict[str, str], ...]] = field(
        default_factory=lambda:
            {
                'gaze': (
                    {
                        'resource': 'jgae7/',
                        'filename': 'sbsat_csvs.zip',
                        'md5': 'a6ef1fb0ecced683cdb489c3bd3e1a5c',
                    },
                ),
                'precomputed_events': (
                    {
                        'resource': '18sat_fixfinal.csv',
                        'filename': '18sat_fixfinal.csv',
                        'md5': '4cf3212a71e6fc2fbe7041ce7c691927',
                    },
                ),
            },
    )
    extract: dict[str, bool] = field(
        default_factory=lambda: {
            'gaze': True, 'precomputed_events': False,
        },
    )

    experiment: Experiment = Experiment(
        screen_width_px=1024,
        screen_height_px=768,
        screen_width_cm=44.5,
        screen_height_cm=42.4,
        distance_cm=70,
        origin='center',
        sampling_rate=1000,
    )

    filename_format: dict[str, str] = field(
        default_factory=lambda:
            {
                'gaze': r'msd{subject_id:d}.csv',
                'precomputed_events': '18sat_fixfinal.csv',
            },
    )

    filename_format_schema_overrides: dict[str, dict[str, type]] = field(
        default_factory=lambda:
            {
                'gaze': {'subject_id': int},
                'precomputed_events': {},
            },
    )

    trial_columns: list[str] = field(
        default_factory=lambda: [
            'book_name',
            'screen_id',
        ],
    )

    time_column: str = 'time'

    time_unit: str = 'ms'

    pixel_columns: list[str] = field(default_factory=lambda: ['x_left', 'y_left'])

    column_map: dict[str, str] = field(default_factory=lambda: {})

    custom_read_kwargs: dict[str, dict[str, Any]] = field(
        default_factory=lambda:
            {
                'gaze': {
                    'separator': '\t',
                    'columns': ['time', 'book_name', 'screen_id', 'x_left', 'y_left', 'pupil_left'],
                    'schema_overrides': {
                        'time': pl.Int64,
                        'book_name': pl.Utf8,
                        'screen_id': pl.Int64,
                        'x_left': pl.Float32,
                        'y_left': pl.Float32,
                        'pupil_left': pl.Float32,
                    },
                },
                'precomputed_events': {'separator': ','},
            },
    )
