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
"""This module provides an interface to the pymovements example eyelink toy dataset."""
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
class ToyDatasetEyeLink(DatasetDefinition):
    """Example toy dataset with EyeLink data.

    This dataset includes monocular eye tracking data from a single participants in a single
    session. Eye movements are recorded at a sampling frequency of 1000 Hz using an EyeLink Portable
    Duo video-based eye tracker and are provided as pixel coordinates.

    The participant is instructed to read a single text and some JuDo trials.

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
    Initialize your :py:class:`~pymovements.Dataset` object with the
    :py:class:`~pymovements.ToyDataset` definition:

    >>> import pymovements as pm
    >>>
    >>> dataset = pm.Dataset("ToyDatasetEyeLink", path='data/ToyDatasetEyeLink')

    Download the dataset resources resources:

    >>> dataset.download()# doctest: +SKIP

    Load the data into memory:

    >>> dataset.load()# doctest: +SKIP
    """
    # pylint: disable=similarities
    # The DatasetDefinition child classes potentially share code chunks for definitions.

    name: str = 'ToyDatasetEyeLink'

    mirrors: tuple[str, ...] = (
        'http://github.com/aeye-lab/pymovements-toy-dataset-eyelink/zipball/',
    )

    resources: tuple[dict[str, str], ...] = (
        {
            'resource': 'a970d090588542dad745297866e794ab9dad8795/',
            'filename': 'pymovements-toy-dataset-eyelink.zip',
            'md5': 'b1d426751403752c8a154fc48d1670ce',
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

    filename_format: str = r'subject_{subject_id:d}_session_{session_id:d}.asc'

    filename_format_dtypes: dict[str, type] = field(
        default_factory=lambda: {
            'subject_id': int,
            'session_id': int,
        },
    )

    column_map: dict[str, str] = field(default_factory=lambda: {})

    custom_read_kwargs: dict[str, Any] = field(
        default_factory=lambda: {
            'patterns': [
                {
                    'pattern': 'SYNCTIME_READING_SCREEN',
                    'column': 'task',
                    'value': 'reading',
                },
                {
                    'pattern': 'SYNCTIME_JUDO',
                    'column': 'task',
                    'value': 'judo',
                },
                {
                    'pattern': ('READING[.]STOP', 'JUDO[.]STOP'),
                    'column': 'task',
                    'value': None,
                },

                r'TRIALID (?P<trial_id>\d+)',
                {
                    'pattern': 'TRIAL_RESULT',
                    'column': 'trial_id',
                    'value': None,
                },

                r'SYNCTIME_READING_SCREEN_(?P<screen_id>\d+)',
                {
                    'pattern': 'READING[.]STOP',
                    'column': 'screen_id',
                    'value': None,
                },

                r'SYNCTIME.P(?P<point_id>\d+)',
                {
                    'pattern': r'P\d[.]STOP',
                    'column': 'point_id',
                    'value': None,
                },
            ],
            'schema': {'trial_id': pl.Int64},
        },
    )
