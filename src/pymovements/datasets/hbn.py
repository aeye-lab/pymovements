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
"""This module provides an interface to the HBN dataset."""
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
class HBN(DatasetDefinition):
    """HBN dataset :cite:p:`HBN`.

    This dataset consists of recordings from children
    watching four different age-appropriate videos: (1) an
    educational video clip (Fun with Fractals), (2) a short animated
    film (The Present), (3) a short clip of an animated film (Despicable Me),
    and (4) a trailer for a feature-length movie (Diary of a Wimpy Kid).
    The eye gaze was recorded at a sampling rate of 120 Hz.

    Check the respective paper for details :cite:p:`HBN`.

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
    :py:class:`~pymovements.HBN` definition:

    >>> import pymovements as pm
    >>>
    >>> dataset = pm.Dataset("HBN", path='data/HBN')

    Download the dataset resources resources:

    >>> dataset.download()# doctest: +SKIP

    Load the data into memory:

    >>> dataset.load()# doctest: +SKIP
    """

    # pylint: disable=similarities
    # The PublicDatasetDefinition child classes potentially share code chunks for definitions.

    name: str = 'HBN'

    mirrors: tuple[str, ...] = (
        'https://files.osf.io/v1/resources/qknuv/providers/osfstorage/',
    )

    resources: tuple[dict[str, str], ...] = (
        {
            'resource': '651190031e76a453918a9971',
            'filename': 'data.zip',
            'md5': '2c523e911022ffc0eab700e34e9f7f30',
        },
    )

    experiment: Experiment = Experiment(
        screen_width_px=800,
        screen_height_px=600,
        screen_width_cm=33.8,
        screen_height_cm=27.0,
        distance_cm=63.5,
        origin='center',
        sampling_rate=120,
    )

    filename_format: str = r'{subject_id:12}_{video_id}.csv'

    filename_format_dtypes: dict[str, type] = field(
        default_factory=lambda: {
            'subject_id': str,
            'video_id': str,
        },
    )

    trial_columns: list[str] = field(default_factory=lambda: ['video_id'])

    time_column: str = 'time'

    pixel_columns: list[str] = field(default_factory=lambda: ['x_pix', 'y_pix'])

    column_map: dict[str, str] = field(default_factory=lambda: {})

    custom_read_kwargs: dict[str, Any] = field(
        default_factory=lambda: {
            'separator': ',',
            'columns': ['time', 'x_pix', 'y_pix'],
            'dtypes': {
                'time': pl.Int64,
                'x_pix': pl.Float32,
                'y_pix': pl.Float32,
            },
        },
    )
