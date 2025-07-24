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
"""Provides a definition for the ChineseReading dataset."""
from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import Any

from pymovements.dataset.dataset_definition import DatasetDefinition


@dataclass
class ChineseReading(DatasetDefinition):
    """ChineseReading dataset :cite:p:`ChineseReading`.

    This dataset includes eye tracking data from more than 300 participants recorded in a single
    session. Precomputed events and word-level reading measures are reported.

    Each participant is instructed to read several sentences.

    Check the respective paper for details :cite:p:`ChineseReading`.

    Attributes
    ----------
    name: str
        The name of the dataset.

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

    trial_columns: list[str]
            The name of the trial columns in the input data frame. If the list is empty or None,
            the input data frame is assumed to contain only one trial. If the list is not empty,
            the input data frame is assumed to contain multiple trials and the transformation
            methods will be applied to each trial separately.

    column_map: dict[str, str]
        The keys are the columns to read, the values are the names to which they should be renamed.

    custom_read_kwargs: dict[str, dict[str, Any]]
        If specified, these keyword arguments will be passed to the file reading function.

    Examples
    --------
    Initialize your :py:class:`~pymovements.dataset.Dataset` object with the
    :py:class:`~pymovements.datasets.ChineseReading` definition:

    >>> import pymovements as pm
    >>>
    >>> dataset = pm.Dataset("ChineseReading", path='data/ChineseReading')

    Download the dataset resources:

    >>> dataset.download()# doctest: +SKIP

    Load the data into memory:

    >>> dataset.load()# doctest: +SKIP
    """

    # pylint: disable=similarities
    # The PublicDatasetDefinition child classes potentially share code chunks for definitions.

    name: str = 'ChineseReading'

    has_files: dict[str, bool] = field(
        default_factory=lambda: {
            'gaze': False,
            'precomputed_events': True,
            'precomputed_reading_measures': True,
        },
    )

    resources: ResourceDefinitions = field(
        default_factory=lambda: ResourceDefinitions.from_dict(
            {
                'precomputed_events': [
                    {
                        'resource':
                        'https://files.osf.io/v1/resources/94wue/'
                        'providers/osfstorage/6253cb37840dd726e75c831a',
                        'filename': 'Raw Data.txt',
                        'md5': None,  # type: ignore
                    },
                ],
                'precomputed_reading_measures': [
                    {
                        'resource':
                        'https://files.osf.io/v1/resources/94wue/providers/osfstorage/?zip=',
                        'filename': 'chinese_reading_measures.zip',
                        'md5': None,  # type: ignore
                    },
                ],
            },
        ),
    )

    filename_format: dict[str, str] = field(
        default_factory=lambda:
            {
                'precomputed_events': 'Raw Data.txt',
                'precomputed_reading_measures': r'{measure_type:s} Measures.xlsx',
            },
    )

    filename_format_schema_overrides: dict[str, dict[str, type]] = field(
        default_factory=lambda:
            {
                'precomputed_events': {},
                'precomputed_reading_measures': {},
            },
    )

    trial_columns: list[str] = field(
        default_factory=lambda: [
            'Subject',
            'Sentence_ID',
        ],
    )

    column_map: dict[str, str] = field(default_factory=lambda: {})

    custom_read_kwargs: dict[str, dict[str, Any]] = field(
        default_factory=lambda:
            {
                'precomputed_events': {'separator': '\t'},
                'precomputed_reading_measures': {'sheet_name': 'Sheet 1'},
            },
    )
