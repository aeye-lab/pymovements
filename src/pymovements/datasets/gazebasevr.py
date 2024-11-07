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
"""Provides a definition for the GazeBase dataset."""
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
class GazeBaseVR(DatasetDefinition):
    """GazeBaseVR dataset :cite:p:`GazeBaseVR`.

    This dataset includes binocular plus an additional cyclopian eye tracking data from 407
    participants captured over a 26-month period. Participants attended up to 3 rounds during this
    time frame, with each round consisting of two contiguous sessions.

    Eye movements are recorded at a sampling frequency of 250 Hz a using SensoMotoric
    Instrument’s (SMI’s) tethered ET VR head-mounted display based on the
    HTC Vive (hereon called the ET-HMD) eye tracker and are provided as
    positional data in degrees of visual angle.

    In each of the two sessions per round, participants are instructed to complete a series of
    tasks, a vergence task (VRG), a smooth pursuit task (PUR), a video viewing task (VID),
    a reading task (TEX), and a random saccade task (RAN).

    Check the respective paper for details :cite:p:`GazeBaseVR`.

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

    position_columns: list[str]
        The name of the dva position columns in the input data frame. These columns will be
        nested into the column ``position``. If the list is empty or None, the nested
        ``position`` column will not be created.

    column_map: dict[str, str]
        The keys are the columns to read, the values are the names to which they should be renamed.

    custom_read_kwargs: dict[str, dict[str, Any]]
        If specified, these keyword arguments will be passed to the file reading function.


    Examples
    --------
    Initialize your :py:class:`~pymovements.PublicDataset` object with the
    :py:class:`~pymovements.GazeBaseVR` definition:

    >>> import pymovements as pm
    >>>
    >>> dataset = pm.Dataset("GazeBaseVR", path='data/GazeBaseVR')

    Download the dataset resources:

    >>> dataset.download()# doctest: +SKIP

    Load the data into memory:

    >>> dataset.load()# doctest: +SKIP
    """

    # pylint: disable=similarities
    # The PublicDatasetDefinition child classes potentially share code chunks for definitions.

    name: str = 'GazeBaseVR'

    has_files: dict[str, bool] = field(
        default_factory=lambda: {
            'gaze': True,
            'precomputed_events': False,
            'precomputed_reading_measures': False,
        },
    )
    mirrors: dict[str, tuple[str, ...]] = field(
        default_factory=lambda: {
            'gaze': (
                'https://figshare.com/ndownloader/files/',
            ),
        },
    )

    resources: dict[str, tuple[dict[str, str], ...]] = field(
        default_factory=lambda: {
            'gaze': (
                {
                    'resource': '38844024',
                    'filename': 'gazebasevr.zip',
                    'md5': '048c04b00fd64347375cc8d37b451a22',
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
        distance_cm=55,
        origin='center',
        sampling_rate=250,
    )

    filename_format: dict[str, str] = field(
        default_factory=lambda: {
            'gaze': (
                r'S_{round_id:1d}{subject_id:d}'
                r'_S{session_id:d}'
                r'_{task_name}.csv'
            ),
        },
    )

    filename_format_schema_overrides: dict[str, dict[str, type]] = field(
        default_factory=lambda: {
            'gaze': {
                'round_id': int,
                'subject_id': int,
                'session_id': int,
            },
        },
    )

    trial_columns: list[str] = field(
        default_factory=lambda: ['round_id', 'subject_id', 'session_id', 'task_name'],
    )

    time_column: str = 'n'

    time_unit: str = 'ms'

    position_columns: list[str] = field(default_factory=lambda: ['lx', 'ly', 'rx', 'ry', 'x', 'y'])

    column_map: dict[str, str] = field(
        default_factory=lambda: {
            'xT': 'x_target_pos',
            'yT': 'y_target_pos',
        },
    )

    custom_read_kwargs: dict[str, dict[str, Any]] = field(
        default_factory=lambda: {
            'gaze': {
                'schema_overrides': {
                    'n': pl.Float32,
                    'x': pl.Float32,
                    'y': pl.Float32,
                    'lx': pl.Float32,
                    'ly': pl.Float32,
                    'rx': pl.Float32,
                    'ry': pl.Float32,
                    'xT': pl.Float32,
                    'yT': pl.Float32,
                    'zT': pl.Float32,
                    'clx': pl.Float32,
                    'cly': pl.Float32,
                    'clz': pl.Float32,
                    'crx': pl.Float32,
                    'cry': pl.Float32,
                    'crz': pl.Float32,
                },
            },
        },
    )
