# Copyright (c) 2023 The pymovements Project Authors
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
"""This module provides an interface to the pymovements example toy dataset."""
from __future__ import annotations

from pathlib import Path

from pymovements.datasets.public_dataset import PublicDataset
from pymovements.gaze.experiment import Experiment


class ToyDataset(PublicDataset):
    """Example toy dataset.

    This dataset includes monocular eye tracking data from a single participants in a single
    session. Eye movements are recorded at a sampling frequency of 1000 Hz using an EyeLink Portable
    Duo video-based eye tracker and are provided as pixel coordinates.

    The participant is instructed to read 4 texts with 5 screens each.

    Examples
    --------
    Change to ``download=True`` and ``extract=True`` for downloading and extracting the dataset.

    >>> dataset = ToyDataset(
    ...     root='data/',
    ...     download=False,
    ...     extract=False,
    ...     remove_finished=False,
    ... )
    >>> dataset.load()  # doctest: +SKIP
    """
    # pylint: disable=similarities
    # The PublicDataset child classes potentially share code chunks for definitions.

    _mirrors = [
        'http://github.com/aeye-lab/pymovements-toy-dataset/zipball/',
    ]

    _resources = [
        {
            'resource': '6cb5d663317bf418cec0c9abe1dde5085a8a8ebd/',
            'filename': 'pymovements-toy-dataset.zip',
            'md5': '4da622457637a8181d86601fe17f3aa8',
        },
    ]

    _experiment = Experiment(
        screen_width_px=1280,
        screen_height_px=1024,
        screen_width_cm=38,
        screen_height_cm=30.2,
        distance_cm=68,
        origin='lower left',
        sampling_rate=1000,
    )

    _filename_regex = r'trial_(?P<text_id>\d+)_(?P<page_id>\d+).csv'

    _filename_regex_dtypes = {
        'text_id': int,
        'page_id': int,
    }

    _column_map = {
        'timestamp': 'time',
        'x': 'x_right_pix',
        'y': 'y_right_pix',
    }

    _read_csv_kwargs = {
        'sep': '\t',
        'columns': list(_column_map.keys()),
        'new_columns': list(_column_map.values()),
        'null_values': '-32768.00',
    }

    def __init__(
            self,
            root: str | Path,
            download: bool = False,
            extract: bool = False,
            remove_finished: bool = False,
            dataset_dirname: str = 'ToyDataset',
            downloads_dirname: str = 'downloads',
            raw_dirname: str = 'raw',
            preprocessed_dirname: str = 'preprocessed',
            events_dirname: str = 'events',
    ):
        """Initialize the pymovements example toy dataset object.

        If desired, dataset resources are downloaded with ``download=True`` and extracted with
        ``extract=True``. To save space on your device you can remove the archive files after
        successful extraction with ``remove_finished=True``.

        Downloaded archives are automatically checked for integrity by comparing MD5 checksums.

        You can set up a custom directory structure by populating the particular dirname attributes.
        See :py:attr:`~pymovements.dataset.PublicDataset.dataset_dirname`,
        :py:attr:`~pymovements.dataset.PublicDataset.raw_dirname`,
        :py:attr:`~pymovements.dataset.PublicDataset.preprocessed_dirname` and
        :py:attr:`~pymovements.dataset.PublicDataset.events_dirname` and
        :py:attr:`~pymovements.dataset.PublicDataset.downloads_dirname` for details.

        Parameters
        ----------
        root : str, Path
            Path to the root directory of the dataset.
        download : bool
            Download all dataset resources.
        extract : bool
            Extract dataset archive files.
        remove_finished : bool
            Remove archive files after extraction.
        dataset_dirname : str, optional
            Dataset directory name under root path. Can be `.` if dataset is located in root path.
            Default: `.`
        downloads_dirname : str, optional
            Name of directory to store downloaded data.Default: `downloads`
        raw_dirname ; str, optional
            Name of directory under dataset path that contains raw data. Can be `.` if raw data is
            located in dataset path. We advise the user to keep the original raw data separate from
            the preprocessed / event data. Default: `raw`
        preprocessed_dirname : str, optional
            Name of directory under dataset path that will be used to store preprocessed data. We
            advise the user to keep the preprocessed data separate from the original raw data.
            Default: `preprocessed`
        events_dirname : str, optional
            Name of directory under dataset path that will be used to store event data. We advise
            the user to keep the event data separate from the original raw data. Default: `events`
        """
        super().__init__(
            root=root,
            download=download,
            extract=extract,
            remove_finished=remove_finished,
            experiment=self._experiment,
            filename_regex=self._filename_regex,
            filename_regex_dtypes=self._filename_regex_dtypes,
            custom_read_kwargs=self._read_csv_kwargs,
            dataset_dirname=dataset_dirname,
            downloads_dirname=downloads_dirname,
            raw_dirname=raw_dirname,
            preprocessed_dirname=preprocessed_dirname,
            events_dirname=events_dirname,
        )

    @property
    def path(self) -> Path:
        """The path to the dataset directory.

        The dataset path points to the dataset directory under the root path. Per default the
        dataset directory name is equal to the class name.

        Example
        -------
        The default behaviour is to locate the data set in a directory under the root path with the
        same name as the class name:
        >>> dataset = ToyDataset(root='/path/to/all/your/datasets')
        >>> dataset.path  # doctest: +SKIP
        Path('/path/to/all/your/datasets/ToyDataset')

        You can specify an explicit dataset directory name:
        >>> dataset = ToyDataset(
        ...     root='/path/to/all/your/datasets',
        ...     dataset_dirname='toy',
        ... )
        >>> dataset.path  # doctest: +SKIP
        Path('/path/to/all/your/datasets/toy')

        You can specify to use the root path to be the actual dataset directory:
        >>> dataset = ToyDataset(root='/path/to/toy/dataset', dataset_dirname='.')
        >>> dataset.path  # doctest: +SKIP
        Path('/path/to/toy/dataset')
        """
        return super().path
