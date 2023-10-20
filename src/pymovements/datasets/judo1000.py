# Copyright (c) 2022-2023 The pymovements Project Authors
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
"""This module provides an interface to the JuDo1000 dataset."""
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
class JuDo1000(DatasetDefinition):
    """JuDo1000 dataset :cite:p:`JuDo1000`.

    This dataset includes binocular eye tracking data from 150 participants in four sessions with an
    interval of at least one week between two sessions. Eye movements are recorded at a sampling
    frequency of 1000 Hz using an EyeLink Portable Duo video-based eye tracker and are provided as
    pixel coordinates. Participants are instructed to watch a random jumping dot on a computer
    screen.

    Check the respective `repository <https://osf.io/5zpvk/>`_ for details.

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
    :py:class:`~pymovements.JuDo1000` definition:

    >>> import pymovements as pm
    >>>
    >>> dataset = pm.Dataset("JuDo1000", path='data/JuDo1000')

    Download the dataset resources resources:

    >>> dataset.download()# doctest: +SKIP

    Load the data into memory:

    >>> dataset.load()# doctest: +SKIP
    """

    # pylint: disable=similarities
    # The PublicDatasetDefinition child classes potentially share code chunks for definitions.

    name: str = 'JuDo1000'

    mirrors: tuple[str, ...] = (
        'https://osf.io/download/',
    )

    resources: tuple[dict[str, str], ...] = (
        {
            'resource': '4wy7s/',
            'filename': 'JuDo1000.zip',
            'md5': 'b8b9e5bb65b78d6f2bd260451cdd89f8',
        },
    )

    experiment: Experiment = Experiment(
        screen_width_px=1280,
        screen_height_px=1024,
        screen_width_cm=38,
        screen_height_cm=30.2,
        distance_cm=68,
        origin='lower left',
        sampling_rate=1000,
    )

    filename_format: str = r'{subject_id:d}_{session_id:d}.csv'

    filename_format_dtypes: dict[str, type] = field(
        default_factory=lambda: {
            'subject_id': int,
            'session_id': int,
        },
    )

    trial_columns: list[str] = field(
        default_factory=lambda: ['subject_id', 'session_id', 'trial_id'],
    )

    time_column: str = 'time'
    pixel_columns: list[str] = field(
        default_factory=lambda: [
            'x_left', 'y_left', 'x_right', 'y_right',
        ],
    )

    column_map: dict[str, str] = field(
        default_factory=lambda: {
            'trialId': 'trial_id',
            'pointId': 'point_id',
        },
    )

    custom_read_kwargs: dict[str, Any] = field(
        default_factory=lambda: {
            'dtypes': {
                'trialId': pl.Int32,
                'pointId': pl.Int32,
                'time': pl.Int64,
                'x_left': pl.Float32,
                'y_left': pl.Float32,
                'x_right': pl.Float32,
                'y_right': pl.Float32,
            },
            'separator': '\t',
        },
    )
