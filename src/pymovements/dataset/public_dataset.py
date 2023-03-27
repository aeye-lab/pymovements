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
from pathlib import Path
from urllib.error import URLError

from pymovements.dataset.dataset import Dataset
from pymovements.dataset.dataset_definition import DatasetDefinition
from pymovements.dataset.dataset_paths import DatasetPaths
from pymovements.utils.archives import extract_archive
from pymovements.utils.downloads import download_file


class PublicDataset(Dataset, metaclass=ABCMeta):
    """Extends the :py:class:`~pymovements.Dataset` base class with functionality for downloading
    and extracting public datasets.

    To implement this abstract base class for a new dataset, the attributes/properties `_mirrors`
    and `_resources` must be implemented.
    """

    def __init__(
            self,
            definition: str | DatasetDefinition | type[DatasetDefinition],
            path: str | Path | DatasetPaths | None,
    ):
        """Initialize the public dataset object.

        Parameters
        ----------
        definition : str, DatasetDefinition
            Dataset definition to initialize dataset with.
        path : DatasetPaths, optional
            Path to the directory of the dataset. You can set up a custom directory structure by
            passing a :py:class:`~pymovements.DatasetPaths` instance.
        """
        super().__init__(definition=definition, path=path)

    def download(self, *, extract: bool = True, remove_finished: bool = False) -> Dataset:
        """Download dataset resources.

        This downloads all resources of the dataset. Per default this also extracts all archives
        into :py:meth:`Dataset.paths.raw`,
        To save space on your device you can remove the archive files after
        successful extraction with ``remove_finished=True``.

        If a corresponding file already exists in the local system, its checksum is calculated and
        checked against the expected checksum.
        Downloading will be evaded if the integrity of the existing file can be verified.
        If the existing file does not match the expected checksum it is overwritten with the
        downloaded new file.

        Parameters
        ----------
        extract : bool
            Extract dataset archive files.
        remove_finished : bool
            Remove archive files after extraction.

        Raises
        ------
        AttributeError
            If number of mirrors or number of resources specified for dataset is zero.
        RuntimeError
            If downloading a resource failed for all given mirrors.

        Returns
        -------
        PublicDataset
            Returns self, useful for method cascading.
        """
        if len(self.definition.mirrors) == 0:
            raise AttributeError('number of mirrors must not be zero to download dataset')

        if len(self.definition.resources) == 0:
            raise AttributeError('number of resources must not be zero to download dataset')

        self.paths.raw.mkdir(parents=True, exist_ok=True)

        for resource in self.definition.resources:
            success = False

            for mirror in self.definition.mirrors:

                url = f'{mirror}{resource["resource"]}'

                try:
                    download_file(
                        url=url,
                        dirpath=self.paths.downloads,
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

        if extract:
            self.extract(remove_finished=remove_finished)

        return self

    def extract(self, remove_finished: bool = False) -> Dataset:
        """Extract downloaded dataset archive files.

        Parameters
        ----------
        remove_finished : bool
            Remove archive files after extraction.

        Returns
        -------
        PublicDataset
            Returns self, useful for method cascading.
        """
        self.paths.raw.mkdir(parents=True, exist_ok=True)

        for resource in self.definition.resources:
            extract_archive(
                source_path=self.paths.downloads / resource['filename'],
                destination_path=self.paths.raw,
                recursive=True,
                remove_finished=remove_finished,
            )
        return self
