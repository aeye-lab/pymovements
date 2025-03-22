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
"""Provides a definition for the ETDD70 dataset."""
from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import Any

from pymovements.dataset.dataset_definition import DatasetDefinition


@dataclass
class ETDD70(DatasetDefinition):
    """ETDD70 dataset :cite:p:`ETDD70`.

    This dataset includes binocular eye tracking data from 70 Czech children age 9-10.
    Eye movements are recorded at a sampling frequency of 250 Hz eye tracker and
    precomputed events are reported.

    Each participant is instructed to read three texts:
        - Task called Syllables contains 90 syllables arranged in a 9 x 10 matrix
        - Task called MeaningfulText consists of a passage about
          a young boy who watches a squirrel from his window.
        - Task called PseudoText comprises fictional, meaningless words.

    Check the respective paper for details :cite:p:`ETDD70`.

    Attributes
    ----------
    name: str
        The name of the dataset.

    has_files: dict[str, bool]
        Indicate whether the dataset contains 'gaze', 'precomputed_events', and
        'precomputed_reading_measures'.

    mirrors: dict[str, list[str]]
        A list of mirrors of the dataset. Each entry must be of type `str` and end with a '/'.

    resources: dict[str, list[dict[str, str]]]
        A list of dataset gaze_resources. Each list entry must be a dictionary with the following
        keys:
        - `resource`: The url suffix of the resource. This will be concatenated with the mirror.
        - `filename`: The filename under which the file is saved as.
        - `md5`: The MD5 checksum of the respective file.

    extract: dict[str, bool]
        Decide whether to extract the data.

    filename_format: dict[str, str]
        Regular expression which will be matched before trying to load the file. Namedgroups will
        appear in the `fileinfo` dataframe.

    filename_format_schema_overrides: dict[str, dict[str, type]]
        If named groups are present in the `filename_format`, this makes it possible to cast
        specific named groups to a particular datatype.

    custom_read_kwargs: dict[str, dict[str, Any]]
        If specified, these keyword arguments will be passed to the file reading function.

    Examples
    --------
    Initialize your :py:class:`~pymovements.dataset.Dataset` object with the
    :py:class:`~pymovements.datasets.ETDD70` definition:

    >>> import pymovements as pm
    >>>
    >>> dataset = pm.Dataset("ETDD70", path='data/ETDD70')

    Download the dataset resources:

    >>> dataset.download()# doctest: +SKIP

    Load the data into memory:

    >>> dataset.load()# doctest: +SKIP
    """

    # pylint: disable=similarities
    # The PublicDatasetDefinition child classes potentially share code chunks for definitions.

    name: str = 'ETDD70'

    has_files: dict[str, bool] = field(
        default_factory=lambda: {
            'gaze': True,
            'precomputed_events': True,
            'precomputed_reading_measures': False,
        },
    )
    mirrors: dict[str, list[str]] = field(
        default_factory=lambda:
            {
                'gaze': [
                    'https://zenodo.org/api/records/13332134/files-archive',
                ],
                'precomputed_events': [
                    'https://zenodo.org/api/records/13332134/files-archive',
                ],
            },
    )
    resources: dict[str, list[dict[str, str]]] = field(
        default_factory=lambda:
            {
                'gaze': [
                    {
                        'resource': '',
                        'filename': 'edd_raw.zip',
                        'md5': None,  # type: ignore
                    },
                ],
                'precomputed_events': [
                    {
                        'resource': '',
                        'filename': 'edd_fix.zip',
                        'md5': None,  # type: ignore
                    },
                ],
            },
    )
    extract: dict[str, bool] = field(
        default_factory=lambda: {
            'gaze': True,
            'precomputed_events': True,
        },
    )

    filename_format: dict[str, str] = field(
        default_factory=lambda:
            {
                'gaze': r'Subject_{subject_id:d}_{task:s}_raw.csv',
                'precomputed_events': r'Subject_{subject_id:d}_{task:s}_fixations.csv',
            },
    )

    filename_format_schema_overrides: dict[str, dict[str, type]] = field(
        default_factory=lambda:
            {
                'gaze': {},
                'precomputed_events': {},
            },
    )

    custom_read_kwargs: dict[str, dict[str, Any]] = field(
        default_factory=lambda:
            {
                'gaze': {},
                'precomputed_events': {},
            },
    )
