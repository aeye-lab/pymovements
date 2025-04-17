from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pymovements.dataset.dataset_definition import DatasetDefinition


@dataclass
class MECOL1W1(DatasetDefinition):
    """MECOL1W1 dataset :cite:p:`MECOL1W1`.

    This dataset includes eye tracking data from several participants in a single
    session. The participants read several paragraphs of texts.

    The participant is instructed to read texts and answer questions.

    Check the respective paper for details :cite:p:`MECOL1W1`.

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
    :py:class:`~pymovements.datasets.MECOL1W1` definition:

    >>> import pymovements as pm
    >>>
    >>> dataset = pm.Dataset("MECOL1W1", path='data/MECOL1W1')

    Download the dataset resources:

    >>> dataset.download()# doctest: +SKIP

    Load the data into memory:

    >>> dataset.load()# doctest: +SKIP
    """

    # pylint: disable=similarities

    name: str = 'MECOL1W1'

    has_files: dict[str, bool] = field(
        default_factory=lambda: {
            'gaze': False,
            'precomputed_events': True,
            'precomputed_reading_measures': True,
        },
    )

    mirrors: dict[str, list[str]] = field(
        default_factory=lambda: {
            'precomputed_events': [
                'https://osf.io/download/',
            ],
            'precomputed_reading_measures': [
                'https://osf.io/download/',
            ],
        },
    )

    resources: dict[str, list[dict[str, str]]] = field(
        default_factory=lambda: {
            'precomputed_events': [
                {
                    'resource': '67dc6027920cab9abae48b83/',
                    'filename': 'joint_l1_fixation_version1.3.rda',
                    'md5': '3c969a930a71cd62c67b936426dd079b',
                },
            ],
            'precomputed_reading_measures': [
                {
                    'resource': 'n5pvh/',
                    'filename': 'sentence_data_version1.3.csv',
                    'md5': '609f82b6f45b7c98a0769c6ce14ee6e9',
                },
            ],
        },
    )

    extract: dict[str, bool] = field(
        default_factory=lambda: {
            'precomputed_events': False,
            'precomputed_reading_measures': False,
        },
    )

    filename_format: dict[str, str] = field(
        default_factory=lambda: {
            'precomputed_events': 'joint_l1_fixation_version1.3.rda',
            'precomputed_reading_measures': 'sentence_data_version1.3.csv',
        },
    )

    filename_format_schema_overrides: dict[str, dict[str, type]] = field(
        default_factory=lambda: {
            'precomputed_events': {},
            'precomputed_reading_measures': {},
        },
    )

    trial_columns: list[str] = field(
        default_factory=lambda: [
            'uniform_id',
            'itemid',
        ],
    )

    column_map: dict[str, str] = field(default_factory=lambda: {})

    custom_read_kwargs: dict[str, dict[str, Any]] = field(
        default_factory=lambda: {
            'precomputed_events': {'r_dataframe_key': 'joint.fix'},
            'precomputed_reading_measures': {},
        },
    )
