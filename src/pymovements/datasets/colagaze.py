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
"""Provides a definition for the CoLAGaze dataset."""
from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import Any

from pymovements.dataset.dataset_definition import DatasetDefinition
from pymovements.gaze.experiment import Experiment


@dataclass
class CoLAGaze(DatasetDefinition):
    """CoLAGaze dataset :cite:p:`CoLAGaze`.

    This dataset includes eye-tracking data from native speakers of English reading
    sentences from the CoLA dataset. Eye movements are recorded at a sampling frequency of 2,000 Hz
    using an EyeLink 1000 eye tracker and are provided as pixel coordinates.

    Check the respective paper for details :cite:p:`CoLAGaze`.

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

    experiment: Experiment
        The experiment definition.

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
    :py:class:`~pymovements.datasets.CoLAGaze` definition:

    >>> import pymovements as pm
    >>>
    >>> dataset = pm.Dataset("CoLAGaze", path='data/CoLAGaze')

    Download the dataset resources:

    >>> dataset.download()# doctest: +SKIP

    Load the data into memory:

    >>> dataset.load()# doctest: +SKIP
    """

    # pylint: disable=similarities
    # The PublicDatasetDefinition child classes potentially share code chunks for definitions.

    name: str = 'CoLAGaze'

    long_name: str = 'Corpus of Eye Movements for Linguistic Acceptability'

    has_files: dict[str, bool] = field(
        default_factory=lambda: {
            'gaze': True,
            'precomputed_events': True,
            'precomputed_reading_measures': True,
        },
    )

    resources: dict[str, list[dict[str, str]]] = field(
        default_factory=lambda: {
            'gaze': [
                {
                    'resource':
                    'https://files.au-1.osf.io/v1/resources/gj2uk/providers/osfstorage/'
                    '67e14ce0f392601163f33215/?view_only=a8ac6e0091e64d0a81d5b1fdec9bab6e&zip=',
                    'filename': 'raw_data.zip',
                    'md5': None,  # type: ignore
                },
            ],
            'precomputed_events': [
                {
                    'resource':
                    'https://files.au-1.osf.io/v1/resources/gj2uk/providers/osfstorage/'
                    '67e14ce0f392601163f33215/?view_only=a8ac6e0091e64d0a81d5b1fdec9bab6e&zip=',
                    'filename': 'fixations.zip',
                    'md5': None,  # type: ignore
                },
            ],
            'precomputed_reading_measures': [
                {
                    'resource':
                    'https://files.au-1.osf.io/v1/resources/gj2uk/providers/osfstorage/'
                    '67e14ce0f392601163f33215/?view_only=a8ac6e0091e64d0a81d5b1fdec9bab6e&zip=',
                    'filename': 'measures.zip',
                    'md5': None,  # type: ignore
                },
            ],
        },
    )

    experiment: Experiment = field(
        default_factory=lambda: Experiment(
            screen_width_px=1280,
            screen_height_px=1024,
            screen_width_cm=54.37,
            screen_height_cm=30.26,
            distance_cm=60,
            origin='bottom left',
            sampling_rate=2000,
        ),
    )

    filename_format: dict[str, str] = field(
        default_factory=lambda: {
            'gaze': '{subject_id:d}.asc',
            'precomputed_events': 'fixations_report_{subject_id:d}.csv',
            'precomputed_reading_measures': 'raw_measures_for_features{subject_id:d}.csv',
        },
    )

    filename_format_schema_overrides: dict[str, dict[str, type]] = field(
        default_factory=lambda: {
            'gaze': {'subject_id': int},
            'precomputed_events': {'subject_id': int},
            'precomputed_reading_measures': {'subject_id': int},
        },
    )

    custom_read_kwargs: dict[str, dict[str, Any]] = field(
        default_factory=lambda: {
            'gaze': {},
            'precomputed_events': {},
            'precomputed_reading_measures': {},
        },
    )
