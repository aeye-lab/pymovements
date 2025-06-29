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
"""Provides a definition for the MouseCursor dataset."""
from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import Any

import polars as pl

from pymovements.dataset.dataset_definition import DatasetDefinition
from pymovements.gaze.experiment import Experiment


@dataclass
class MouseCursor(DatasetDefinition):
    """MouseCursor dataset :cite:p:`MouseCursor`.

    The paper presents a dataset for comparing two types of eye tracking:
    smooth (vestibulo-ocular reflex or VOR-based) and saccadic eye tracking.
    Data were collected using a head-mounted infrared camera system that adjusted
    the mouse cursor based on pupil position.
    The experiments involved two participants completing tasks to either position the
    cursor within target areas or move it to a target location as quickly as possible.
    The dataset is intended to help researchers evaluate and compare eye tracking methods
    without needing to build the system themselves.

    Attributes
    ----------
    name: str
        The name of the dataset.

    long_name: str
        The entire name of the dataset.

    has_files: dict[str, bool]
        Indicate whether the dataset contains 'gaze', 'precomputed_events', and
        'precomputed_reading_measures'.

    resources: dict[str, list[dict[str, str]]]
        A tuple of dataset gaze_resources. Each list entry must be a dictionary with the following
        keys:
        - `resource`: The url suffix of the resource. This will be concatenated with the mirror.
        - `filename`: The filename under which the file is saved as.
        - `md5`: The MD5 checksum of the respective file.

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
    Initialize your :py:class:`~pymovements.datasets.Dataset` object with the
    :py:class:`~pymovements.datasets.MouseCursor` definition:

    >>> import pymovements as pm
    >>>
    >>> dataset = pm.Dataset("MouseCursor", path='data/MouseCursor')

    Download the dataset resources:

    >>> dataset.download()# doctest: +SKIP

    Load the data into memory:

    >>> dataset.load()# doctest: +SKIP
    """

    # pylint: disable=similarities
    # All DatasetDefinition classes potentially share large code chunks.

    name: str = 'MouseCursor'

    long_name: str = 'Mouse Cursor dataset'

    has_files: dict[str, bool] = field(
        default_factory=lambda: {
            'gaze': True,
            'precomputed_events': False,
            'precomputed_reading_measures': False,
        },
    )

    resources: dict[str, list[dict[str, str]]] = field(
        default_factory=lambda: {
            'gaze': [
                {
                    'resource': 'https://ars.els-cdn.com/content/image/1-s2.0-S2352340921000160-mmc1.zip',  # noqa: E501 # pylint: disable=line-too-long
                    'filename': 'mousecursor.zip',
                    'md5': '7885e8fd44f14f02f60e9f62431aea63',
                },
            ],
        },
    )

    experiment: Experiment = field(
        default_factory=lambda: Experiment(
            # page 4: middle of screen is 960/540 => 1920/1080
            screen_width_px=1920,
            screen_height_px=1080,
            # page 5: calculation
            screen_width_cm=52.99,
            screen_height_cm=29.81,
            distance_cm=50,
            origin='upper left',
            # page 5: 334 ms intervals of tracking gaze (PlayStation-Eye Camera)
            sampling_rate=3,
        ),
    )

    filename_format: dict[str, str] = field(
        default_factory=lambda: {
            'gaze': r'Experiment {experiment_id:d}.csv',
        },
    )

    filename_format_schema_overrides: dict[str, dict[str, type]] = field(
        default_factory=lambda: {
            'gaze': {
                'experiment_id': int,
            },
        },
    )

    trial_columns: list[str] = field(default_factory=lambda: ['Trial', 'Participant'])

    time_column: str = 'Time'

    time_unit: str = 'ms'

    pixel_columns: list[str] = field(default_factory=lambda: ['x', 'y'])

    column_map: dict[str, str] = field(default_factory=lambda: {})

    custom_read_kwargs: dict[str, dict[str, Any]] = field(
        default_factory=lambda: {
            'gaze': {
                'schema_overrides': {
                    'Tracking': pl.Utf8,
                    'Trial': pl.Int64,
                    'Measurement': pl.Int64,
                    'ExactTime': pl.Utf8,
                    'Time': pl.Float32,
                    'x': pl.Float32,
                    'y': pl.Float32,
                    'Participant': pl.Int64,
                },
            },
        },
    )
