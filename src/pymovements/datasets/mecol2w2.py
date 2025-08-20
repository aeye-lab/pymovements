# Copyright (c) 2025 The pymovements Project Authors
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
"""Provides a definition for the MECOL2W2 dataset."""
from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import Any

from pymovements.dataset.dataset_definition import DatasetDefinition
from pymovements.dataset.resources import ResourceDefinitions


@dataclass
class MECOL2W2(DatasetDefinition):
    """MECOL2W2 dataset :cite:p:`MECOL2W2`.

    This dataset includes eye tracking data from several participants in a single
    session. The participants read several paragraphs of texts.

    The participant is instructed to read texts and answer questions.

    Check the respective paper for details :cite:p:`MECOL2W2`.

    Attributes
    ----------
    name: str
        The name of the dataset.

    long_name: str
        The entire name of the dataset.

    resources: ResourceDefinitions
        A list of dataset gaze_resources. Each list entry must be a dictionary with the following
        keys:
        - `resource`: The url suffix of the resource. This will be concatenated with the mirror.
        - `filename`: The filename under which the file is saved as.
        - `md5`: The MD5 checksum of the respective file.

    filename_format: dict[str, str] | None
        Regular expression which will be matched before trying to load the file. Namedgroups will
        appear in the `fileinfo` dataframe.

    filename_format_schema_overrides: dict[str, dict[str, type]] | None
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
    :py:class:`~pymovements.datasets.MECOL2W2` definition:

    >>> import pymovements as pm
    >>>
    >>> dataset = pm.Dataset("MECOL2W2", path='data/MECOL2W2')

    Download the dataset resources:

    >>> dataset.download()# doctest: +SKIP

    Load the data into memory:

    >>> dataset.load()# doctest: +SKIP
    """

    # pylint: disable=similarities

    name: str = 'MECOL2W2'

    long_name: str = 'Multilingual Eye-tracking Corpus second language reader second wave'

    resources: ResourceDefinitions = field(
        default_factory=lambda: ResourceDefinitions.from_dicts(
            [
                {
                    'content': 'precomputed_events',
                    'url': 'https://osf.io/download/677e22d30788b62a2dd10a52/',
                    'filename': 'joint_fix_trimmed_L2_wave2.rda',
                    'md5': 'f7eaf80ac5916d79351419fd4f1da2f6',
                    'filename_pattern': 'joint_fix_trimmed_L2_wave2.rda',
                },
                {
                    'content': 'precomputed_reading_measures',
                    'url': 'https://osf.io/download/tma46/',
                    'filename': 'joint_data_trimmed_L2_wave2_2025_01_03.rda',
                    'md5': '87077697db7f09172a3431615397a5e9',
                    'filename_pattern': 'joint_data_trimmed_L2_wave2_2025_01_03.rda',
                },
            ],
        ),
    )

    filename_format: dict[str, str] | None = None

    filename_format_schema_overrides: dict[str, dict[str, type]] | None = None

    trial_columns: list[str] = field(
        default_factory=lambda: [
            'uniform_id',
            'itemid',
        ],
    )

    column_map: dict[str, str] = field(default_factory=lambda: {})

    custom_read_kwargs: dict[str, dict[str, Any]] = field(
        default_factory=lambda: {
            'precomputed_events': {'r_dataframe_key': 'joint.fix.l2_w2'},
            'precomputed_reading_measures': {'r_dataframe_key': 'joint.data'},
        },
    )
