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
"""This module provides an interface to the GazeBase dataset."""
from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import Any

from pymovements.datasets.public_dataset import PublicDatasetDefinition
from pymovements.datasets.public_dataset import register_public_dataset
from pymovements.gaze.experiment import Experiment


@dataclass
@register_public_dataset
class GazeBase(PublicDatasetDefinition):
    """GazeBase dataset :cite:p:`GazeBase`.

    This dataset includes monocular (left eye) eye tracking data from 322 participants captured over
    a period of 37 months. Participants attended up to 9 rounds during this time frame, with each
    round consisting of two contiguous sessions.

    Eye movements are recorded at a sampling frequency of 1000 Hz using an EyeLink 1000 video-based
    eye tracker and are provided as positional data in degrees of visual angle.

    In each of the two sessions per round, participants are instructed to complete a series of
    tasks, including a fixation task (FIX), a horizontal saccade task (HSS), a random saccade task
    (RAN), a reading task (TEX), two free viewing video tasks (VD1 and VD2) and a gaze-driven gaming
    task (BLG).

    Check the respective paper for details :cite:p:`GazeBase`.

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

    filename_regex : str
        Regular expression which will be matched before trying to load the file. Namedgroups will
        appear in the `fileinfo` dataframe.

    filename_regex_dtypes : dict[str, type], optional
        If named groups are present in the `filename_regex`, this makes it possible to cast specific
        named groups to a particular datatype.

    column_map : dict[str, str]
        The keys are the columns to read, the values are the names to which they should be renamed.

    custom_read_kwargs : dict[str, Any], optional
        If specified, these keyword arguments will be passed to the file reading function.

    Examples
    --------
    Initialize your :py:class:`~pymovements.PublicDataset` object with the
    :py:class:`~pymovements.GazeBase` definition:

    >>> import pymovements as pm
    >>>
    >>> dataset = pm.PublicDataset("GazeBase", root='data/')

    Download the dataset resources resources:

    >>> dataset.download()# doctest: +SKIP

    Load the data into memory:

    >>> dataset.load()# doctest: +SKIP
    """
    # pylint: disable=similarities
    # The PublicDatasetDefinition child classes potentially share code chunks for definitions.

    name: str = 'GazeBase'

    mirrors: tuple[str] = (
        'https://figshare.com/ndownloader/files/',
    )

    resources: tuple[dict[str, str]] = (
        {
            'resource': '27039812',
            'filename': 'GazeBase_v2_0.zip',
            'md5': 'cb7eb895fb48f8661decf038ab998c9a',
        },
    )

    experiment: Experiment = Experiment(
        screen_width_px=1680,
        screen_height_px=1050,
        screen_width_cm=47.4,
        screen_height_cm=29.7,
        distance_cm=55,
        origin='lower left',
        sampling_rate=1000,
    )

    filename_regex: str = (
        r'S_(?P<round_id>\d)(?P<subject_id>\d+)'
        r'_S(?P<session_id>\d+)'
        r'_(?P<task_name>.+).csv'
    )

    filename_regex_dtypes: dict[str, type] = field(
        default_factory=lambda: {
            'round_id': int,
            'subject_id': int,
            'session_id': int,
        },
    )

    column_map: dict[str, str] = field(
        default_factory=lambda: {
            'n': 'time',
            'x': 'x_left_pos',
            'y': 'y_left_pos',
            'val': 'validity',
            'xT': 'x_target_pos',
            'yT': 'y_target_pos',
        },
    )

    custom_read_kwargs: dict[str, Any] = field(default_factory=dict)
