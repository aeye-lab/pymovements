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
"""Provides a definition for the OneStop dataset."""
from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import Any

from pymovements.dataset.dataset_definition import DatasetDefinition


@dataclass
class OneStop(DatasetDefinition):
    """OneStop dataset :cite:p:`OneStop`.

    OneStop Eye Movements (in short OneStop) is an English corpus of eye movements
    in reading with 360 L1 participants, 2.6 million word tokens and 152 hours of
    eye tracking data recorded with an EyeLink 1000 Plus eye tracker.
    OneStop comprises four sub-corpora with eye movement recordings from paragraph reading.

    To filter the data by reading regime or trial type, use the following column values:

    For ordinary reading trials, set question_preview to False.
    For information seeking trials, set question_preview to True.
    To exclude repeated reading trials, set repeated_reading_trial to False.
    To include only repeated reading trials, set repeated_reading_trial to True.
    To exclude practice trials, set practice_trial to False.

    For more information please consult :cite:p:`OneStop`.

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
        A list of dataset gaze_resources. Each list entry must be a dictionary with the following
        keys:
        - `resource`: The url suffix of the resource. This will be concatenated with the mirror.
        - `filename`: The filename under which the file is saved as.
        - `md5`: The MD5 checksum of the respective file.

    filename_format: dict[str, str]
        Regular expression which will be matched before trying to load the file. Namedgroups will
        appear in the `fileinfo` dataframe.

    filename_format_schema_overrides: dict[str, dict[str, type]]
        If named groups are present in the `filename_format`, this makes it possible to cast
        specific named groups to a particular datatype.

    custom_read_kwargs: dict[str, Any]
        If specified, these keyword arguments will be passed to the file reading function.

    Examples
    --------
    Initialize your :py:class:`~pymovements.dataset.Dataset` object with the
    :py:class:`~pymovements.datasets.OneStop` definition:

    >>> import pymovements as pm
    >>>
    >>> dataset = pm.Dataset("OneStop", path='data/OneStop')

    Download the dataset resources:

    >>> dataset.download()# doctest: +SKIP

    Load the data into memory:

    >>> dataset.load()# doctest: +SKIP
    """

    # pylint: disable=similarities
    # The PublicDatasetDefinition child classes potentially share code chunks for definitions.

    name: str = 'OneStop'

    long_name: str = 'OneStop: A 360-Participant English Eye Tracking Dataset with Different '\
        'Reading Regimes'

    has_files: dict[str, bool] = field(
        default_factory=lambda: {
            'gaze': False,
            'precomputed_events': True,
            'precomputed_reading_measures': True,
        },
    )

    resources: dict[str, list[dict[str, str]]] = field(
        default_factory=lambda: {
            'precomputed_events': [
                {
                    'resource':
                    'https://osf.io/download/dq935/',
                    'filename': 'fixations_Paragraph.csv.zip',
                    'md5': '3d3b6a3794a50e174e025f43735674bd',
                },
            ],
            'precomputed_reading_measures': [
                {
                    'resource': 'https://osf.io/download/4ajc8/',
                    'filename': 'ia_Paragraph.csv.zip',
                    'md5': '9b9548e49efdc7dbf63d4f3a5dc3af22',
                },
            ],
        },
    )

    filename_format: dict[str, str] = field(
        default_factory=lambda: {
            'precomputed_events': 'fixations_Paragraph.csv',
            'precomputed_reading_measures': 'ia_Paragraph.csv',
        },
    )

    filename_format_schema_overrides: dict[str, dict[str, type]] = field(
        default_factory=lambda: {
            'precomputed_events': {},
            'precomputed_reading_measures': {},
        },
    )

    custom_read_kwargs: dict[str, Any] = field(
        default_factory=lambda: {
            'precomputed_events': {'null_values': '.'},
            'precomputed_reading_measures': {'null_values': '.'},
        },
    )
