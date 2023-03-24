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
"""This module provides an interface to the pymovements example toy dataset."""
from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field

from pymovements.datasets.public_dataset import PublicDatasetDefinition
from pymovements.datasets.public_dataset import register_public_dataset
from pymovements.gaze.experiment import Experiment


@dataclass
@register_public_dataset
class ToyDataset(PublicDatasetDefinition):
    """Example toy dataset.

    This dataset includes monocular eye tracking data from a single participants in a single
    session. Eye movements are recorded at a sampling frequency of 1000 Hz using an EyeLink Portable
    Duo video-based eye tracker and are provided as pixel coordinates.

    The participant is instructed to read 4 texts with 5 screens each.

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
    :py:class:`~pymovements.ToyDataset` definition:

    >>> import pymovements as pm
    >>>
    >>> dataset = pm.PublicDataset("ToyDataset", root='data/')

    Download the dataset resources resources:

    >>> dataset.download()# doctest: +SKIP

    Load the data into memory:

    >>> dataset.load()# doctest: +SKIP
    """
    # pylint: disable=similarities
    # The PublicDatasetDefinition child classes potentially share code chunks for definitions.

    name: str = 'ToyDataset'

    mirrors: tuple[str, ...] = (
        'http://github.com/aeye-lab/pymovements-toy-dataset/zipball/',
    )

    resources: tuple[dict[str, str], ...] = (
        {
            'resource': '6cb5d663317bf418cec0c9abe1dde5085a8a8ebd/',
            'filename': 'pymovements-toy-dataset.zip',
            'md5': '4da622457637a8181d86601fe17f3aa8',
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

    filename_regex: str = r'trial_(?P<text_id>\d+)_(?P<page_id>\d+).csv'

    filename_regex_dtypes: dict[str, type] = field(
        default_factory=lambda: {
            'text_id': int,
            'page_id': int,
        },
    )

    column_map: dict[str, str] = field(
        default_factory=lambda: {
            'timestamp': 'time',
            'x': 'x_right_pix',
            'y': 'y_right_pix',
        },
    )

    custom_read_kwargs: dict[str, str] = field(
        default_factory=lambda: {
            'separator': '\t',
            'null_values': '-32768.00',
        },
    )
