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
"""This module provides an interface to the JuDo1000 dataset."""
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
    """PoTeC dataset :cite:p:`PoTeC`.

    The Potsdam Textbook Corpus (PoTeC) is a corpus of eye-tracking-while-reading data where
    participants (N=75) read a series of German short texts taken from college level textbooks
    of physics and biology. The experiments were conducted within a 2x2 fully-crossed factorial
    design with the reader’s expertise (advanced vs beginner) and major (physics vs biology) as
    factors. Reading comprehension was assessed using text comprehension questions. Moreover,
    background questions that required additional knowledge beyond the presented text tested the
    general domain knowledge.
    The repository contains the eye-movement data (1000 Hz) as well as the stimulus text data
    with extensive linguistic feature annotations at the sub-lexical, lexical und supra-lexical
    level. Therefore, the PoTeC is ideal for studying cognitive processes related to sentence
    comprehension at all linguistic levels (e.g. lexical, syntactic, discourse) as well as
    higher-level text comprehension.

    Check the respective `repository <https://osf.io/dn5hp/>`_ for details.

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
    :py:class:`~pymovements.PoTeC` definition:

    >>> import pymovements as pm
    >>>
    >>> dataset = pm.Dataset("PoTeC", path='data/PoTeC')

    Download the dataset resources resources:

    >>> dataset.download()# doctest: +SKIP

    Load the data into memory:

    >>> dataset.load()# doctest: +SKIP
    """

    # pylint: disable=similarities
    # The PublicDatasetDefinition child classes potentially share code chunks for definitions.

    name: str = 'PoTeC'

    mirrors: tuple[str, ...] = (
        'https://osf.io/download/',
    )

    resources: tuple[dict[str, str], ...] = (
        {
            'resource': 'tgd9q/',
            'filename': 'PoTeC.zip',
            'md5': '7780904bf7b18ba7d30a811174750db3',
        },
    )

    experiment: Experiment = Experiment(
        screen_width_px=1680,
        screen_height_px=1050,
        screen_width_cm=47.5,
        screen_height_cm=30,
        distance_cm=65,
        origin='lower left',
        sampling_rate=1000,
    )

    filename_format: str = r'reader{subject_id:d}_{text_id}_raw_data.tsv'

    filename_format_dtypes: dict[str, type] = field(
        default_factory=lambda: {
            'subject_id': int,
            'text_id': str,
        },
    )

    trial_columns: list[str] = field(
        default_factory=lambda: ['subject_id', 'text_id'],
    )

    time_column: str = 'time'
    pixel_columns: list[str] = field(
        default_factory=lambda: [
            'x', 'y',
        ],
    )

    custom_read_kwargs: dict[str, Any] = field(
        default_factory=lambda: {
            'dtypes': {
                'time': pl.Int64,
                'x': pl.Float32,
                'y': pl.Float32,
                'pupil_diameter': pl.Float32,
            },
            'separator': '\t',
        },
    )
