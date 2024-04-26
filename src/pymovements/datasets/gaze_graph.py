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
"""Provides a definition for the GazeGraph dataset."""
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
class GazeGraph(DatasetDefinition):
    """GazeGraph dataset :cite:p:`GazeGraph`.

    The dataset is collected from eight subjects (four female and four male,
    aged between 24 and 35) using the Pupil Core eye tracker. During data collection,
    the subjects wear the eye tracker and sit in front of the computer screen
    (a 34-inch display) at a distance of approximately 50cm. We conduct the
    manufacturer's default on-screen five-points calibration for each of
    the subjects.
    Note that we have done only one calibration per subject, and the subjects
    can move their heads and upper bodies freely during the experiment.
    The gaze is recorded at a 30Hz sampling rate.

    Check the respective paper for details :cite:p:`GazeGraph`.

    Attributes
    ----------
    name : str
        The name of the dataset.

    mirrors : tuple[str, ...]
        A tuple of mirrors of the dataset. Each entry must be of type `str` and end with a '/'.

    resources : tuple[dict[str, str], ...]
        A tuple of dataset resources. Each list entry must be a dictionary with the following keys:
        - `resource`: The url suffix of the resource. This will be concatenated with the mirror.
        - `filename`: The filename under which the file is saved as.
        - `md5`: The MD5 checksum of the respective file.

    experiment : Experiment
        The experiment definition.

    filename_format : str
        Regular expression which will be matched before trying to load the file. Namedgroups will
        appear in the `fileinfo` dataframe.

    filename_format_dtypes : dict[str, type], optional
        If named groups are present in the `filename_format`, this makes it possible to cast
        specific named groups to a particular datatype.

    column_map : dict[str, str]
        The keys are the columns to read, the values are the names to which they should be renamed.

    custom_read_kwargs : dict[str, Any], optional
        If specified, these keyword arguments will be passed to the file reading function.

    Examples
    --------
    Initialize your :py:class:`~pymovements.PublicDataset` object with the
    :py:class:`~pymovements.GazeGraph` definition:

    >>> import pymovements as pm
    >>>
    >>> dataset = pm.Dataset("GazeGraph", path='data/GazeGraph')

    Download the dataset resources:

    >>> dataset.download()# doctest: +SKIP

    Load the data into memory:

    >>> dataset.load()# doctest: +SKIP
    """

    # pylint: disable=similarities
    # The PublicDatasetDefinition child classes potentially share code chunks for definitions.

    name: str = 'GazeGraph'

    mirrors: tuple[str, ...] = (
        'https://codeload.github.com/GazeGraphResource/GazeGraph/zip/refs/heads/',
    )

    resources: tuple[dict[str, str], ...] = (
        {
            'resource': 'master',
            'filename': 'gaze_graph_data.zip',
            'md5': '181f4b79477cee6e0267482d989610b0',
        },
    )

    # no information about the resolution and screen size given. only 34-inch monitor
    experiment: Experiment = Experiment(
        screen_width_px=3440,
        screen_height_px=1440,
        screen_width_cm=79.375,
        screen_height_cm=34.0106,
        distance_cm=50,
        origin='center',
        sampling_rate=30,
    )

    filename_format: str = r'P{subject_id}_{task}.csv'

    filename_format_dtypes: dict[str, type] = field(
        default_factory=lambda: {
            'subject_id': int,
            'task': str,
        },
    )

    trial_columns: list[str] = field(default_factory=lambda: ['task'])

    time_column: Any = None

    time_unit: Any = None

    pixel_columns: list[str] = field(default_factory=lambda: ['x', 'y'])

    column_map: dict[str, str] = field(default_factory=lambda: {})

    custom_read_kwargs: dict[str, Any] = field(
        default_factory=lambda: {
            'separator': ',',
            'has_header': False,
            'new_columns': ['x', 'y'],
            'dtypes': [pl.Float32, pl.Float32],
        },
    )
