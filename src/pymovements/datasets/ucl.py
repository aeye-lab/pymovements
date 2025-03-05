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
"""Provides a definition for the UCL dataset."""
from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import Any

from pymovements.dataset.dataset_definition import DatasetDefinition
from pymovements.dataset.dataset_library import register_dataset


@dataclass
@register_dataset
class UCL(DatasetDefinition):
    """UCL dataset :cite:p:`UCL`.

    UCL is a dataset of word-by-word reading times collected through
    self-paced reading and eye-tracking experiments to evaluate
    computational psycholinguistic models of English sentence comprehension.
    361 sentences from narrative sources, ensuring they were understandable without context,
    and recorded reading times from participants using both methods.

    For more details check out the original paper :cite:p:`UCL`.

    Attributes
    ----------
    name: str
        The name of the dataset.

    has_files: dict[str, bool]
        Indicate whether the dataset contains 'gaze', 'precomputed_events', and
        'precomputed_reading_measures'.

    mirrors: dict[str, tuple[str, ...]]
        A tuple of mirrors of the dataset. Each entry must be of type `str` and end with a '/'.

    resources: dict[str, tuple[dict[str, str], ...]]
        A tuple of dataset gaze_resources. Each list entry must be a dictionary with the following
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

    trial_columns: list[str]
            The name of the trial columns in the input data frame. If the list is empty or None,
            the input data frame is assumed to contain only one trial. If the list is not empty,
            the input data frame is assumed to contain multiple trials and the transformation
            methods will be applied to each trial separately.

    pixel_columns: list[str]
        The name of the pixel position columns in the input data frame. These columns will be
        nested into the column ``pixel``. If the list is empty or None, the nested ``pixel``
        column will not be created.

    column_map: dict[str, str]
        The keys are the columns to read, the values are the names to which they should be renamed.

    custom_read_kwargs: dict[str, Any]
        If specified, these keyword arguments will be passed to the file reading function.

    Examples
    --------
    Initialize your :py:class:`~pymovements.dataset.Dataset` object with the
    :py:class:`~pymovements.datasets.UCL` definition:

    >>> import pymovements as pm
    >>>
    >>> dataset = pm.Dataset("UCL", path='data/UCL')

    Download the dataset resources:

    >>> dataset.download()# doctest: +SKIP

    Load the data into memory:

    >>> dataset.load()# doctest: +SKIP
    """

    # pylint: disable=similarities
    # The PublicDatasetDefinition child classes potentially share code chunks for definitions.

    name: str = 'UCL'

    has_files: dict[str, bool] = field(
        default_factory=lambda: {
            'gaze': False,
            'precomputed_events': True,
            'precomputed_reading_measures': True,
        },
    )
    mirrors: dict[str, tuple[str, ...]] = field(
        default_factory=lambda: {
            'precomputed_events': (
                'https://static-content.springer.com/esm/'
                'art%3A10.3758%2Fs13428-012-0313-y/MediaObjects/',
            ),
            'precomputed_reading_measures': (
                'https://static-content.springer.com/esm/'
                'art%3A10.3758%2Fs13428-012-0313-y/MediaObjects/',
            ),
        },
    )
    resources: dict[str, tuple[dict[str, str], ...]] = field(
        default_factory=lambda: {
            'precomputed_events': (
                {
                    'resource': '13428_2012_313_MOESM1_ESM.zip',
                    'filename': '13428_2012_313_MOESM1_ESM.zip',
                    'md5': '77e3c0cacccb0a074a55d23aa8531ca5',
                },
            ),
            'precomputed_reading_measures': (
                {
                    'resource': '13428_2012_313_MOESM1_ESM.zip',
                    'filename': '13428_2012_313_MOESM1_ESM.zip',
                    'md5': '77e3c0cacccb0a074a55d23aa8531ca5',
                },
            ),
        },
    )

    extract: dict[str, bool] = field(
        default_factory=lambda: {
            'precomputed_events': True,
            'precomputed_reading_measures': True,
        },
    )

    filename_format: dict[str, str] = field(
        default_factory=lambda: {
            'precomputed_events': r'eyetracking.fix',
            'precomputed_reading_measures': r'eyetracking.RT',
        },
    )

    filename_format_schema_overrides: dict[str, dict[str, type]] = field(
        default_factory=lambda: {
            'precomputed_events': {},
            'precomputed_reading_measures': {},
        },
    )

    trial_columns: list[str] = field(default_factory=lambda: [])

    pixel_columns: list[str] = field(default_factory=lambda: [])

    column_map: dict[str, str] = field(default_factory=lambda: {})

    custom_read_kwargs: dict[str, Any] = field(
        default_factory=lambda: {
            'precomputed_events': {
                'separator': '\t',
                'null_values': ['NaN'],
            },
            'precomputed_reading_measures': {'separator': '\t'},
        },
    )
