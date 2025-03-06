# Copyright (c) 2023-2025 The pymovements Project Authors
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
"""Provides private functions for downloading and extracting datasets."""
from __future__ import annotations

import shutil
from urllib.error import URLError

from pymovements.dataset.dataset_definition import DatasetDefinition
from pymovements.dataset.dataset_paths import DatasetPaths
from pymovements.utils.archives import extract_archive
from pymovements.utils.downloads import download_file


def download_dataset(
        definition: DatasetDefinition,
        paths: DatasetPaths,
        *,
        extract: bool = True,
        remove_finished: bool = False,
        resume: bool = False,
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
        Checks for correct size of existing files but not integrity. (default: False)
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
    if definition.has_files['gaze']:
        if len(definition.mirrors['gaze']) == 0:
            raise AttributeError('number of mirrors must not be zero to download dataset')

        if len(definition.resources['gaze']) == 0:
            raise AttributeError('number of `gaze` resources must not be zero to download dataset')

        paths.raw.mkdir(parents=True, exist_ok=True)

        for resource in definition.resources['gaze']:
            success = False

            for mirror_idx, mirror in enumerate(definition.mirrors['gaze']):

                url = f'{mirror}{resource["resource"]}'

                try:
                    download_file(
                        url=url,
                        dirpath=paths.downloads,
                        filename=resource['filename'],
                        md5=resource['md5'],
                        verbose=verbose,
                    )
                    success = True

                # pylint: disable=overlapping-except
                except (URLError, OSError, RuntimeError) as error:
                    # Error downloading the resource, try next mirror
                    if mirror_idx < len(definition.mirrors['gaze']) - 1:
                        print(f'Failed to download:\n{error}\nTrying next mirror.')
                    continue

                # downloading the resource was successful, we don't need to try another mirror
                break

            if not success:
                raise RuntimeError(
                    f"downloading resource {resource['resource']} failed for all mirrors.",
                )

    if definition.has_files['precomputed_events']:
        if len(definition.mirrors['precomputed_events']) == 0:
            raise AttributeError(
                'number of precomputed mirrors must not be zero to download dataset',
            )

        if len(definition.resources['precomputed_events']) == 0:
            raise AttributeError(
                'number of `precomputed_events` resources must not be zero to download dataset',
            )

        paths.precomputed_events.mkdir(parents=True, exist_ok=True)

        for resource in definition.resources['precomputed_events']:
            success = False

            for mirror_idx, mirror in enumerate(definition.mirrors['precomputed_events']):

                url = f'{mirror}{resource["resource"]}'

                try:
                    download_file(
                        url=url,
                        dirpath=paths.downloads,
                        filename=resource['filename'],
                        md5=resource['md5'],
                        verbose=verbose,
                    )
                    success = True

                # pylint: disable=overlapping-except
                except (URLError, OSError, RuntimeError) as error:
                    # Error downloading the resource, try next mirror
                    if mirror_idx < len(definition.mirrors['precomputed_events']) - 1:
                        print(f'Failed to download:\n{error}\nTrying next mirror.')
                    continue

                # downloading the resource was successful, we don't need to try another mirror
                break

            if not success:
                raise RuntimeError(
                    f"downloading resource {resource['resource']} "
                    'failed for all mirrors.',
                )

    if definition.has_files['precomputed_reading_measures']:
        if len(definition.mirrors['precomputed_reading_measures']) == 0:
            raise AttributeError(
                'number of `precomputed_reading_measures` mirrors must in datasets definition'
                'not be zero to download dataset',
            )

        if len(definition.resources['precomputed_reading_measures']) == 0:
            raise AttributeError(
                'number of `precomputed_reading_measures` resources in datasets definition'
                'must not be zero to download dataset',
            )

        paths.precomputed_reading_measures.mkdir(parents=True, exist_ok=True)

        for resource in definition.resources['precomputed_reading_measures']:
            success = False

            for mirror_idx, mirror in enumerate(definition.mirrors['precomputed_reading_measures']):

                url = f'{mirror}{resource["resource"]}'

                try:
                    download_file(
                        url=url,
                        dirpath=paths.downloads,
                        filename=resource['filename'],
                        md5=resource['md5'],
                        verbose=verbose,
                    )
                    success = True

                # pylint: disable=overlapping-except
                except (URLError, OSError, RuntimeError) as error:
                    # Error downloading the resource, try next mirror
                    if mirror_idx < len(definition.mirrors['precomputed_reading_measures']) - 1:
                        print(f'Failed to download:\n{error}\nTrying next mirror.')
                    continue

                # downloading the resource was successful, we don't need to try another mirror
                break

            if not success:
                raise RuntimeError(
                    f"downloading resource {resource['resource']} "
                    'failed for all mirrors.',
                )

    if extract:
        extract_dataset(
            definition=definition,
            paths=paths,
            remove_finished=remove_finished,
            resume=resume,
            verbose=verbose,
        )


def extract_dataset(
        definition: DatasetDefinition,
        paths: DatasetPaths,
        *,
        remove_finished: bool = False,
        remove_top_level: bool = True,
        resume: bool = False,
        verbose: int = 1,
) -> None:
    """Extract downloaded dataset archive files.

    Parameters
    ----------
    definition: DatasetDefinition
        The dataset definition.
    paths: DatasetPaths
        The dataset paths.
    remove_finished: bool
        Remove archive files after extraction. (default: False)
    remove_top_level: bool
        If ``True``, remove the top-level directory if it has only one child. (default: True)
    resume: bool
        Resume previous extraction by skipping existing files.
        Checks for correct size of existing files but not integrity. (default: False)
    verbose: int
        Verbosity levels: (1) Print messages for extracting each dataset resource without printing
        messages for recursive archives. (2) Print messages for extracting each dataset resource and
        each recursive archive extract. (default: 1)
    """
    if definition.has_files['gaze'] and definition.extract['gaze']:
        paths.raw.mkdir(parents=True, exist_ok=True)
        for resource in definition.resources['gaze']:
            source_path = paths.downloads / resource['filename']
            destination_path = paths.raw

            extract_archive(
                source_path=source_path,
                destination_path=destination_path,
                recursive=True,
                remove_finished=remove_finished,
                remove_top_level=remove_top_level,
                resume=resume,
                verbose=verbose,
            )

    if definition.has_files['precomputed_events']:
        paths.precomputed_events.mkdir(parents=True, exist_ok=True)
        for resource in definition.resources['precomputed_events']:
            source_path = paths.downloads / resource['filename']
            destination_path = paths.precomputed_events

            if definition.extract['precomputed_events']:
                extract_archive(
                    source_path=source_path,
                    destination_path=destination_path,
                    recursive=True,
                    remove_finished=remove_finished,
                    remove_top_level=remove_top_level,
                    resume=resume,
                    verbose=verbose,
                )
            else:
                shutil.copy(source_path, destination_path / resource['filename'])

    if definition.has_files['precomputed_reading_measures']:
        paths.precomputed_reading_measures.mkdir(parents=True, exist_ok=True)
        for resource in definition.resources['precomputed_reading_measures']:
            source_path = paths.downloads / resource['filename']
            destination_path = paths.precomputed_reading_measures

            if definition.extract['precomputed_reading_measures']:
                extract_archive(
                    source_path=source_path,
                    destination_path=destination_path,
                    recursive=True,
                    remove_finished=remove_finished,
                    remove_top_level=remove_top_level,
                    resume=resume,
                    verbose=verbose,
                )
            else:
                shutil.move(source_path, destination_path / resource['filename'])
