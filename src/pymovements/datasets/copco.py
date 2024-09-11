# Copyright (c) 2022-2024 The pymovements Project Authors
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
"""Provides a definition for the CopCo dataset."""
from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import Any

from pymovements.dataset.dataset_definition import DatasetDefinition
from pymovements.dataset.dataset_library import register_dataset
from pymovements.gaze.experiment import Experiment


@dataclass
@register_dataset
class CopCo(DatasetDefinition):
    """CopCo dataset :cite:p:`CopCoL1Hollenstein`.

    This dataset includes monocular eye tracking data from a single participants in a single
    session. Eye movements are recorded at a sampling frequency of 1,000 Hz using an EyeLink 1000
    eye tracker and are provided as pixel coordinates.

    The participant is instructed to read texts and answer questions.

    The dataset includes the data from three papers:
        the L1 data: :cite:p:`CopCoL1Hollenstein`,
        the L1 data with dylsexia: :cite:p:`CopCoL1DysBjornsdottir`,
        the L2 data: :cite:p:`CopCoL2`,

    Attributes
    ----------
    name : str
        The name of the dataset.

    mirrors : tuple[str, ...]
        A tuple of mirrors of the dataset. Each entry must be of type `str` and end with a '/'.

    resources : tuple[dict[str, str], ...]
        A tuple of dataset gaze_resources. Each list entry must be a dictionary with the following
        keys:
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
    :py:class:`~pymovements.CopCo` definition:

    >>> import pymovements as pm
    >>>
    >>> dataset = pm.Dataset("CopCo", path='data/CopCo')

    Download the dataset resources:

    >>> dataset.download()# doctest: +SKIP

    Load the data into memory:

    >>> dataset.load()# doctest: +SKIP
    """

    # pylint: disable=similarities
    # The PublicDatasetDefinition child classes potentially share code chunks for definitions.

    name: str = 'CopCo'

    has_files: dict[str, bool] = field(
        default_factory=lambda: {
            'gaze': False,
            'precomputed_events': True,
        },
    )
    extract: dict[str, bool] = field(default_factory=lambda: {'precomputed_events': True})
    mirrors: dict[str, tuple[str, ...]] = field(
        default_factory=lambda: {
            'precomputed_events': ('https://files.de-1.osf.io/',),
        },
    )
    resources: dict[str, tuple[dict[str, str | None], ...]] = field(
        default_factory=lambda: {
            'precomputed_events': (
                {
                    'resource':
                    'v1/resources/ud8s5/providers/osfstorage/61e13174c99ebd02df017c14/?zip=',
                    'filename': 'FixationReports.zip',
                    'md5': None,  # type:ignore
                },
            ),
        },
    )

    experiment: Experiment = Experiment(
        screen_width_px=1920,
        screen_height_px=1080,
        screen_width_cm=59.,
        screen_height_cm=33.5,
        distance_cm=85,
        origin='center',
        sampling_rate=1000,
    )

    filename_format: dict[str, str] = field(
        default_factory=lambda: {
            'precomputed_events': r'FIX_report_P{subject_id:d}.txt',
        },
    )

    filename_format_dtypes: dict[str, dict[str, type]] = field(
        default_factory=lambda: {
            'precomputed_events': {},
        },
    )

    trial_columns: list[str] = field(default_factory=lambda: [])

    time_column: str = ''

    time_unit: str = ''

    pixel_columns: list[str] = field(default_factory=lambda: [])

    column_map: dict[str, str] = field(default_factory=lambda: {})

    custom_read_kwargs: dict[str, Any] = field(
        default_factory=lambda: {
            'precomputed_events': {
                'separator': '\t',
                'null_values': '.',
                'quote_char': '"',
            },
        },
    )
