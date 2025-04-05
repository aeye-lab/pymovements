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
"""Utils module for downloading files."""
from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path
from typing import Any
from urllib.error import URLError
from warnings import warn

from tqdm.auto import tqdm

from pymovements._version import get_versions
from pymovements.dataset._archives import _extract_dataset
from pymovements.dataset.dataset_definition import DatasetDefinition
from pymovements.dataset.dataset_paths import DatasetPaths

_USER_AGENT: str = f"pymovements/{get_versions()['version']}"


def _download_dataset(
        definition: DatasetDefinition,
        paths: DatasetPaths,
        *,
        extract: bool = True,
        remove_finished: bool = False,
        resume: bool = True,
        verbose: bool = True,
) -> None:
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
    definition: DatasetDefinition
        The dataset definition.
    paths: DatasetPaths
        The dataset paths.
    extract: bool
        Extract dataset archive files. (default: True)
    remove_finished: bool
        Remove archive files after extraction. (default: False)
    resume: bool
        Resume previous extraction by skipping existing files.
        Checks for correct size of existing files but not integrity. (default: True)
    verbose: bool
        If True, show progress of download and print status messages for integrity checking and
        file extraction. (default: True)

    Raises
    ------
    AttributeError
        If number of mirrors or number of resources specified for dataset is zero.
    RuntimeError
        If downloading a resource failed for all given mirrors.
    """
    if not definition.resources:
        raise AttributeError('resources must be specified to download dataset.')

    for content in ['gaze', 'precomputed_events', 'precomputed_reading_measures']:
        if definition.has_files[content]:
            if not definition.mirrors:
                mirrors = None
            else:
                mirrors = definition.mirrors.get(content, None)

            if not definition.resources[content]:
                raise AttributeError(
                    f"'{content}' resources must be specified to download dataset.")

            _download_resources(
                mirrors=mirrors,
                resources=definition.resources[content],
                target_dirpath=paths.downloads,
                verbose=verbose,
            )

    if extract:
        _extract_dataset(
            definition=definition,
            paths=paths,
            remove_finished=remove_finished,
            resume=resume,
            verbose=verbose,
        )


def _download_resources(
        mirrors: list[str] | tuple[str, ...] | None,
        resources: list[dict[str, str]] | tuple[dict[str, str], ...],
        target_dirpath: Path,
        verbose: bool,
) -> None:
    """Download resources."""
    for resource in resources:
        if not mirrors:
            _download_resource_without_mirrors(resource, target_dirpath, verbose)
        else:
            assert mirrors is not None
            _download_resource_with_mirrors(mirrors, resource, target_dirpath, verbose)


def _download_resource_without_mirrors(
        resource: dict[str, str],
        target_dirpath: Path,
        verbose: bool,
) -> None:
    """Download resouce without mirrors."""
    try:
        _download_file(
            url=resource['resource'],
            dirpath=target_dirpath,
            filename=resource['filename'],
            md5=resource['md5'],
            verbose=verbose,
        )

    # pylint: disable=overlapping-except
    except (URLError, OSError, RuntimeError) as error:
        raise RuntimeError(
            f"downloading resource {resource['resource']} failed.",
        ) from error


def _download_resource_with_mirrors(
        mirrors: list[str] | tuple[str, ...],
        resource: dict[str, str],
        target_dirpath: Path,
        verbose: bool,
) -> None:
    """Download resource with mirrors."""
    success = False

    for mirror_idx, mirror in enumerate(mirrors):

        url = f'{mirror}{resource["resource"]}'

        try:
            _download_file(
                url=url,
                dirpath=target_dirpath,
                filename=resource['filename'],
                md5=resource['md5'],
                verbose=verbose,
            )
            success = True

        # pylint: disable=overlapping-except
        except (URLError, OSError, RuntimeError) as error:
            # Error downloading the resource, try next mirror
            if mirror_idx < len(mirrors) - 1:
                warning = UserWarning(
                    f'Failed to download from mirror {mirror}\nTrying next mirror.',
                )
                warning.__cause__ = error
                warn(warning)
            continue

        # downloading the resource was successful, we don't need to try another mirror
        break

    if not success:
        raise RuntimeError(
            f"downloading resource {resource['resource']} failed for all mirrors.",
        )


def _download_file(
        url: str,
        dirpath: Path,
        filename: str,
        md5: str | None = None,
        *,
        max_redirect_hops: int = 3,
        verbose: bool = True,
) -> Path:
    """Download a file from a URL and place it in root.

    Parameters
    ----------
    url : str
        URL of file to be downloaded.
    dirpath : Path
        Path to directory where file will be saved to.
    filename : str
        Target filename of saved file.
    md5 : str | None
        MD5 checksum of downloaded file. If None, do not check. (default: None)
    max_redirect_hops : int
        Maximum number of redirect hops allowed. (default: 3)
    verbose : bool
        If True, show progress bar and print info messages on downloading file. (default: True)

    Returns
    -------
    Path
        Filepath to downloaded file.

    Raises
    ------
    OSError
        If the download process failed.
    RuntimeError
        If the MD5 checksum of the downloaded file did not match the expected checksum.
    """
    dirpath = dirpath.expanduser()
    dirpath.mkdir(parents=True, exist_ok=True)
    filepath = dirpath / filename

    # check if file is already present locally
    if _check_integrity(filepath, md5):
        if verbose:
            print('Using already downloaded and verified file:', filepath)
        return filepath

    if verbose:
        print(f'Downloading {url} to {filepath}')

    # expand redirect chain if needed
    url = _get_redirected_url(url=url, max_hops=max_redirect_hops)

    # download the file
    try:
        _download_url(url=url, destination=filepath, verbose=verbose)

    except OSError as e:
        if url[:5] == 'https':
            print('Download failed. Trying https -> http instead.')
            url = url.replace('https:', 'http:')

            if verbose:
                print(f'Downloading {url} to {filepath}')
            _download_url(url=url, destination=filepath, verbose=verbose)
        else:
            raise e

    # check integrity of downloaded file
    if verbose:
        print(f'Checking integrity of {filepath.name}')
    if not _check_integrity(filepath=filepath, md5=md5):
        raise RuntimeError(f'File {filepath} not found or download corrupted.')

    return filepath


def _get_redirected_url(url: str, max_hops: int = 3) -> str:
    """Get redirected URL.

    Parameters
    ----------
    url: str
        Initial URL to be requested for redirection.
    max_hops: int
        Maximum number of redirection hops. (default: 3)

    Returns
    -------
    str
        Final URL after all redirections.

    Raises
    ------
    RuntimeError
        If number of redirects exceed `max_hops`.
    """
    initial_url = url
    headers = {'Method': 'HEAD', 'User-Agent': _USER_AGENT}

    for _ in range(max_hops + 1):
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers)) as response:
            if response.url == url or response.url is None:
                return url
            url = response.url

    raise RuntimeError(
        f'Request to {initial_url} exceeded {max_hops} redirects.'
        f' The last redirect points to {url}.',
    )


# have to ignore pylint since tqdm 2.45.0 broke incosistent-mro
# https://github.com/tqdm/tqdm/blob/0bb91857eca0d4aea08f66cf1c8949abe0cd6b7a/tqdm/auto.py#L27
class _DownloadProgressBar(tqdm):  # pylint: disable=inconsistent-mro
    """Progress bar for downloads.

    Provides `update_to(n)` which uses `tqdm.update(delta_n)`.

    Reference: https://github.com/tqdm/tqdm#hooks-and-callbacks

    Parameters
    ----------
    **kwargs : Any
    """

    def __init__(self, **kwargs: Any):
        super().__init__(unit='B', unit_scale=True, unit_divisor=1024, miniters=1, **kwargs)

    def update_to(self, b: int = 1, bsize: int = 1, tsize: int | None = None) -> bool | None:
        """Update progress bar.

        Parameters
        ----------
        b  : int
            Number of blocks transferred so far (default: 1).
        bsize  : int
            Size of each block (in tqdm units) (default: 1).
        tsize  : int | None
            Total size (in tqdm units). If None it remains unchanged (default: None).

        Returns
        -------
        bool | None
            Returns `None` if the update was successful, `False` if the update was skipped
            because the last update was too recent.
        """
        if tsize is not None:
            self.total = tsize
        return self.update(b * bsize - self.n)  # also sets self.n = b * bsize


def _download_url(url: str, destination: Path, verbose: bool = True) -> None:
    """Download file from URL and save to destination.

    Parameters
    ----------
    url : str
        URL of file to be downloaded.
    destination : Path
        Destination path of downloaded file.
    verbose : bool
        If True, show progressbar.
    """
    with _DownloadProgressBar(desc=destination.name, disable=not verbose) as t:
        urllib.request.urlretrieve(url=url, filename=destination, reporthook=t.update_to)
        t.total = t.n


def _check_integrity(filepath: Path, md5: str | None = None) -> bool:
    """Check file integrity by MD5 checksum.

    Parameters
    ----------
    filepath : Path
        Path to file.
    md5: str | None
        Expected MD5 checksum of file. If None, do not check. (default: None)

    Returns
    -------
    bool
        True if file checksum matches passed `md5` or if passed `md5` is None. False if file
        checksum does not match passed `md5` or `filepath` doesn't exist.
    """
    if not filepath.is_file():
        return False
    if md5 is None:
        return True

    # Calculate checksum and check for match.
    file_md5 = _calculate_md5(filepath)
    return file_md5 == md5


def _calculate_md5(filepath: Path, chunk_size: int = 1024 * 1024) -> str:
    """Calculate MD5 checksum.

    Parameters
    ----------
    filepath : Path
        Path to file.
    chunk_size : int
        Byte size of processed chunks. (default: 1024 * 1024)

    Returns
    -------
    str
        Calculated MD5 checksum.
    """
    # Setting the `usedforsecurity` flag does not change anything about the functionality, but
    # indicates that we are not using the MD5 checksum for cryptography.
    # This enables its usage in restricted environments like FIPS without raising an error.
    file_md5 = hashlib.new('md5', usedforsecurity=False)

    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            file_md5.update(chunk)
    return file_md5.hexdigest()
