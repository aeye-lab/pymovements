# Copyright (c) 2023-2023 The pymovements Project Authors
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
from urllib.error import URLError

from pymovements.datasets.dataset import Dataset
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
        root: str,
        download: bool = False,
        extract: bool = False,
        remove_finished: bool = False,
        **kwargs,
    ):
        """Initialize dataset.

        If desired, dataset resources will be downloaded and extracted.

        Parameters
        ----------
        download : bool
            Download all dataset resources.
        extract : bool
            Extract dataset archive files.
        remove_finished : bool
            Remove archive files after extraction.
        """
        super().__init__(root=root, **kwargs)
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
                        dirpath=self.rootpath,
                        filename=resource['filename'],
                        md5=resource['md5'],
                    )
                    success = True

                except URLError as error:
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
                source_path=self.rootpath / resource['filename'],
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
