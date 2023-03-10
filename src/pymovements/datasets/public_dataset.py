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
"""This module provides the abstract base public dataset class."""
from __future__ import annotations

from abc import ABCMeta
from abc import abstractmethod
from pathlib import Path
from typing import Any
from urllib.error import URLError

from pymovements.datasets.dataset import Dataset
from pymovements.gaze.experiment import Experiment
from pymovements.utils.archives import extract_archive
from pymovements.utils.downloads import download_file


class PublicDataset(Dataset, metaclass=ABCMeta):
    """Extends the `Dataset` base class with functionality for downloading and extracting public
    datasets.

    To implement this abstract base class for a new dataset, the attributes/properties `_mirrors`
    and `_resources` must be implemented.
    """

    def __init__(
            self,
            root: str | Path,
            download: bool = False,
            extract: bool = False,
            remove_finished: bool = False,
            experiment: Experiment | None = None,
            filename_regex: str = '.*',
            filename_regex_dtypes: dict[str, type] | None = None,
            custom_read_kwargs: dict[str, Any] | None = None,
            dataset_dirname: str | None = None,
            downloads_dirname: str = 'downloads',
            raw_dirname: str = 'raw',
            preprocessed_dirname: str = 'preprocessed',
            events_dirname: str = 'events',
    ):
        """Initialize the public dataset object.

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
        experiment : Experiment
            The experiment definition.
        filename_regex : str
            Regular expression which needs to be matched before trying to load the file. Named
            groups will appear in the `fileinfo` dataframe.
        filename_regex_dtypes : dict[str, type], optional
            If named groups are present in the `filename_regex`, this makes it possible to cast
            specific named groups to a particular datatype.
        custom_read_kwargs : dict[str, Any], optional
            If specified, these keyword arguments will be passed to the file reading function.
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
        if dataset_dirname is None:
            dataset_dirname = self.__class__.__name__

        super().__init__(
            root=root,
            experiment=experiment,
            filename_regex=filename_regex,
            filename_regex_dtypes=filename_regex_dtypes,
            custom_read_kwargs=custom_read_kwargs,
            dataset_dirname=dataset_dirname,
            raw_dirname=raw_dirname,
            preprocessed_dirname=preprocessed_dirname,
            events_dirname=events_dirname,
        )

        self.downloads_dirname = downloads_dirname

        if download:
            self.download()

        if extract:
            self.extract(remove_finished=remove_finished)

    def download(self) -> None:
        """Download dataset.

        Raises
        ------
        RuntimeError
            If downloading a resource failed for all given mirrors.
        """
        self.raw_rootpath.mkdir(parents=True, exist_ok=True)

        for resource in self._resources:
            success = False

            for mirror in self._mirrors:

                url = f'{mirror}{resource["resource"]}'

                try:
                    download_file(
                        url=url,
                        dirpath=self.downloads_rootpath,
                        filename=resource['filename'],
                        md5=resource['md5'],
                    )
                    success = True

                # pylint: disable=overlapping-except
                except (URLError, OSError, RuntimeError) as error:
                    print(f'Failed to download (trying next):\n{error}')
                    # downloading the resource, try next mirror
                    continue

                # downloading the resource was successful, we don't need to try another mirror
                break

            if not success:
                raise RuntimeError(
                    f"downloading resource {resource['resource']} failed for all mirrors.",
                )

    def extract(self, remove_finished: bool = False) -> None:
        """Extract dataset archives.

        Parameters
        ----------
        remove_finished : bool
            Remove archive files after extraction.
        """
        self.raw_rootpath.mkdir(parents=True, exist_ok=True)

        for resource in self._resources:
            extract_archive(
                source_path=self.downloads_rootpath / resource['filename'],
                destination_path=self.raw_rootpath,
                recursive=True,
                remove_finished=remove_finished,
            )

    @property
    @abstractmethod
    def _mirrors(self):
        """This attribute/property must provide a list of mirrors of the dataset.

        Each entry should be of type `str` and end with a '/'.
        """

    @property
    @abstractmethod
    def _resources(self):
        """This attribute must provide a list of dataset resources.

        Each list entry should be a dictionary with the following keys:

        - resource: The url suffix of the resource. This will be concatenated with the mirror.
        - filename: The filename under which the file is saved as.
        - md5: The MD5 checksum of the respective file.

        All values should be of type string.
        """

    @property
    def path(self) -> Path:
        """The path to the dataset directory.

        The dataset path points to the dataset directory under the root path. Per default the
        dataset directory name is equal to the class name.

        Example
        -------
        Let's define a custom class first with one mirror and one resource.
        >>> class CustomPublicDataset(PublicDataset):
        ...     _mirrors = ['https://www.example.com/']
        ...     _resources = [
        ...         {
        ...             'resource': 'relative_path_to_resource',
        ...             'filename': 'example_dataset.zip',
        ...             'md5': 'md5md5md5md5md5md5md5md5md5md5md',
        ...         },
        ...     ]

        The default behaviour is to locate the data set in a directory under the root path with the
        same name as the class name:
        >>> dataset = CustomPublicDataset(root='/path/to/all/your/datasets')
        >>> dataset.path  # doctest: +SKIP
        Path('/path/to/all/your/datasets/CustomPublicDataset')

        You can specify an explicit dataset directory name:
        >>> dataset = CustomPublicDataset(
        ...     root='/path/to/all/your/datasets',
        ...     dataset_dirname='custom_dataset',
        ... )
        >>> dataset.path  # doctest: +SKIP
        Path('/path/to/all/your/datasets/custom_dataset')

        You can specify to use the root path to be the actual dataset directory:
        >>> dataset = CustomPublicDataset(root='/path/to/your/dataset', dataset_dirname='.')
        >>> dataset.path  # doctest: +SKIP
        Path('/path/to/your/dataset')
        """
        return super().path

    @property
    def downloads_rootpath(self) -> Path:
        """The path to the directory of the raw data.

        The download path points to the download directory under the root path.

        Example
        -------
        Let's define a custom class first with one mirror and one resource.
        >>> class CustomPublicDataset(PublicDataset):
        ...     _mirrors = ['https://www.example.com/']
        ...     _resources = [
        ...         {
        ...             'resource': 'relative_path_to_resource',
        ...             'filename': 'example_dataset.zip',
        ...             'md5': 'md5md5md5md5md5md5md5md5md5md5md',
        ...         },
        ...     ]

        >>> dataset = CustomPublicDataset(root='/path/to/your/datasets/')
        >>> dataset.downloads_rootpath  # doctest: +SKIP
        Path('/path/to/your/datasets/CustomPublicDataset/downloads')

        You can also explicitely specify the preprocessed directory name. The default is `raw`.
        >>> dataset = CustomPublicDataset(
        ...     root='/path/to/your/datasets/',
        ...     dataset_dirname='your_dataset',
        ...     downloads_dirname='your_raw_data',
        ... )
        >>> dataset.downloads_rootpath  # doctest: +SKIP
        Path('/path/to/your/datasets/your_dataset/your_raw_data')

        If your raw data is not in a separate directory under the root path then you can also
        specify `.` as the directory name. We discourage this and advise the user to keep raw data
        and preprocessed data separated.
        ... dataset = Dataset(
        >>> dataset = CustomPublicDataset(
        ...     root='/path/to/your/datasets/',
        ...     dataset_dirname='your_dataset',
        ...     raw_dirname='.',
        ... )
        >>> dataset.preprocessed_rootpath  # doctest: +SKIP
        Path('/path/to/your/datasets/your_dataset/your_raw_data')
        """
        return self.path / self.downloads_dirname
