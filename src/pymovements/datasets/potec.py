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
"""Provides a definition for the PoTeC dataset."""
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
class PoTeC(DatasetDefinition):
    """PoTeC dataset :cite:p:`potec`.

    The Potsdam Textbook Corpus (PoTeC) is a naturalistic eye-tracking-while-reading
    corpus containing data from 75 participants reading 12 scientific texts.
    PoTeC is the first naturalistic eye-tracking-while-reading corpus that contains
    eye-movements from domain-experts as well as novices in a within-participant
    manipulation: It is based on a 2×2×2 fully-crossed factorial design which includes
    the participants' level of study and the participants' discipline of study as
    between-subject factors and the text domain as a within-subject factor. The
    participants' reading comprehension was assessed by a series of text comprehension
    questions and their domain knowledge was tested by text-independent
    background questions for each of the texts. The materials are annotated for a
    variety of linguistic features at different levels. We envision PoTeC to be used
    for a wide range of studies including but not limited to analyses of expert and
    non-expert reading strategies.

    The corpus and all the accompanying data at all
    stages of the preprocessing pipeline and all code used to preprocess the data are
    made available via `GitHub. <https://github.com/DiLi-Lab/PoTeC>`_

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

    experiment: Experiment
        The experiment definition.

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

    time_column: str
        The name of the timestamp column in the input data frame. This column will be renamed to
        ``time``.

    time_unit: str
        The unit of the timestamps in the timestamp column in the input data frame. Supported
        units are 's' for seconds, 'ms' for milliseconds and 'step' for steps. If the unit is
        'step' the experiment definition must be specified. All timestamps will be converted to
        milliseconds.

    pixel_columns: list[str]
        The name of the pixel position columns in the input data frame. These columns will be
        nested into the column ``pixel``. If the list is empty or None, the nested ``pixel``
        column will not be created.

    custom_read_kwargs: dict[str, dict[str, Any]]
        If specified, these keyword arguments will be passed to the file reading function.

    Examples
    --------
    Initialize your :py:class:`~pymovements.PublicDataset` object with the
    :py:class:`~pymovements.PoTeC` definition:

    >>> import pymovements as pm
    >>>
    >>> dataset = pm.Dataset("PoTeC", path='data/PoTeC')

    Download the dataset resources:

    >>> dataset.download()# doctest: +SKIP

    Load the data into memory:

    >>> dataset.load()# doctest: +SKIP
    """

    # pylint: disable=similarities
    # The PublicDatasetDefinition child classes potentially share code chunks for definitions.

    name: str = 'PoTeC'

    has_files: dict[str, bool] = field(
        default_factory=lambda: {
            'gaze': True,
            'precomputed_events': False,
            'precomputed_reading_measures': False,
        },
    )

    mirrors: dict[str, tuple[str, ...]] = field(
        default_factory=lambda: {
            'gaze': ('https://osf.io/download/',),
        },
    )

    resources: dict[str, tuple[dict[str, str], ...]] = field(
        default_factory=lambda: {
            'gaze': (
                {
                    'resource': 'tgd9q/',
                    'filename': 'PoTeC.zip',
                    'md5': 'cffd45039757c3777e2fd130e5d8a2ad',
                },
            ),
        },
    )

    extract: dict[str, bool] = field(default_factory=lambda: {'gaze': True})

    experiment: Experiment = Experiment(
        screen_width_px=1680,
        screen_height_px=1050,
        screen_width_cm=47.5,
        screen_height_cm=30,
        distance_cm=65,
        origin='upper left',
        sampling_rate=1000,
    )

    filename_format: dict[str, str] = field(
        default_factory=lambda: {
            'gaze': r'reader{subject_id:d}_{text_id}_raw_data.tsv',
        },
    )

    filename_format_schema_overrides: dict[str, dict[str, type]] = field(
        default_factory=lambda: {
            'gaze': {
                'subject_id': int,
                'text_id': str,
            },
        },
    )

    trial_columns: list[str] = field(
        default_factory=lambda: ['subject_id', 'text_id'],
    )

    time_column: str = 'time'

    time_unit: str = 'ms'

    pixel_columns: list[str] = field(
        default_factory=lambda: [
            'x', 'y',
        ],
    )

    custom_read_kwargs: dict[str, dict[str, Any]] = field(
        default_factory=lambda: {
            'gaze': {
                'schema_overrides': {
                    'time': pl.Int64,
                    'x': pl.Float32,
                    'y': pl.Float32,
                    'pupil_diameter': pl.Float32,
                },
                'separator': '\t',
            },
        },
    )
