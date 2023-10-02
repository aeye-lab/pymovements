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
"""This module provides an interface to the GazeOnFaces dataset."""
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
class GazeOnFaces(DatasetDefinition):
    """GazeOnFaces dataset :cite:p:`GazeOnFaces`.

    This dataset includes monocular eye tracking data from single participants in a single
    session. Eye movements are recorded at a sampling frequency of 60 Hz
    using an EyeLink 1000 video-based eye tracker and are provided as pixel coordinates.

    Participants were sat 57 cm away from the screen (19inch LCD monitor,
    screen res=1280×1024, 60 Hz). Recordings of the eye movements of one eye in monocular
    pupil/corneal reflection tracking mode.

    Check the respective paper for details :cite:p:`GazeOnFaces`.

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
    :py:class:`~pymovements.GazeOnFaces` definition:

    >>> import pymovements as pm
    >>>
    >>> dataset = pm.Dataset("GazeOnFaces", path='data/GazeOnFaces')

    Download the dataset resources resources:

    >>> dataset.download()# doctest: +SKIP

    Load the data into memory:

    >>> dataset.load()# doctest: +SKIP
    """

    # pylint: disable=similarities
    # The PublicDatasetDefinition child classes potentially share code chunks for definitions.

    name: str = 'GazeOnFaces'

    mirrors: tuple[str, ...] = (
        'https://uncloud.univ-nantes.fr/index.php/s/',
    )

    resources: tuple[dict[str, str], ...] = (
        {
            'resource': '8KW6dEdyBJqxpmo/download?path=%2F&files=gaze_csv.zip',
            'filename': 'gaze_csv.zip',
            'md5': 'fe219f07c9253cd9aaee6bd50233c034',
        },
    )

    experiment: Experiment = Experiment(
        screen_width_px=1280,
        screen_height_px=1024,
        screen_width_cm=38,
        screen_height_cm=30,
        distance_cm=57,
        origin='center',
        sampling_rate=60,
    )

    filename_format: str = r'gaze_sub{sub_id:d}_trial{trial_id:d}.csv'

    filename_format_dtypes: dict[str, type] = field(
        default_factory=lambda: {
            'sub_id': int,
            'trial_id': int,
        },
    )

    trial_columns: list[str] = field(default_factory=lambda: ['sub_id', 'trial_id'])

    time_column: Any = None

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
