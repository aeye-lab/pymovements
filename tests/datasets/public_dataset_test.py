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

"""Test all functionality in pymovements.datasets.public_dataset."""
import tarfile
from pathlib import Path
from unittest import mock

import pytest

from pymovements.datasets.public_dataset import PublicDataset


@pytest.fixture(
    name='single_archive',
    params=[
        ('gz', 'tar'),
    ],
    autouse=True,
)
def fixture_archive(tmp_path):
    rootpath = tmp_path
    compression, extension = 'gz', 'tar'

    # write tmp filepath
    filepath = rootpath / 'test.file'
    filepath.write_text('test')

    archive_path = rootpath / f'test.{extension}.{compression}'

    with tarfile.TarFile.open(archive_path, f'w:{compression}') as fp:
        fp.add(filepath)

    yield archive_path


@pytest.fixture(name='dataset')
def custom_dataset(tmp_path):
    class CustomPublicDataset(PublicDataset):
        _mirrors = ['https://example.com/', 'https://another_example.com/']
        _resources = [
            {
                'resource': 'test.gz.tar',
                'filename': 'test.gz.tar',
                'md5': '52bbf03a7c50ee7152ccb9d357c2bb30'
            }
        ]

    return CustomPublicDataset(
        root=tmp_path,
        dataset_dirname='',
        downloads_dirname=''
    )


@pytest.mark.parametrize(
    'init_kwargs, expected_paths',
    [
        pytest.param(
            {'root': '/data/set/path'},
            {
                'root': Path('/data/set/path/'),
                'path': Path('/data/set/path/CustomPublicDataset'),
                'download': Path('/data/set/path/CustomPublicDataset/downloads'),
            },
        ),
        pytest.param(
            {'root': '/data/set/path', 'dataset_dirname': '.'},
            {
                'root': Path('/data/set/path/'),
                'path': Path('/data/set/path/'),
                'download': Path('/data/set/path/downloads'),
            },
        ),
        pytest.param(
            {'root': '/data/set/path', 'dataset_dirname': 'dataset'},
            {
                'root': Path('/data/set/path/'),
                'path': Path('/data/set/path/dataset'),
                'download': Path('/data/set/path/dataset/downloads'),
            },
        ),
        pytest.param(
            {'root': '/data/set/path', 'downloads_dirname': 'custom_download_dirname'},
            {
                'root': Path('/data/set/path/'),
                'path': Path('/data/set/path/CustomPublicDataset'),
                'download': Path('/data/set/path/CustomPublicDataset/custom_download_dirname'),
            },
        ),
    ],
)
def test_paths(init_kwargs, expected_paths):
    class CustomPublicDataset(PublicDataset):
        _mirrors = ['https://www.example.com/']
        _resources = [
            {
                'resource': 'relative_path_to_resource',
                'filename': 'example_dataset.zip',
                'md5': 'md5md5md5md5md5md5md5md5md5md5md',
            },
        ]

    dataset = CustomPublicDataset(**init_kwargs)

    assert dataset.root == expected_paths['root']
    assert dataset.path == expected_paths['path']
    assert dataset.downloads_rootpath == expected_paths['download']


@mock.patch('pymovements.datasets.public_dataset.download_file')
def test_dataset_download_fails(mock_download_file, tmp_path, dataset):
    mock_download_file.side_effect = OSError()

    with pytest.raises(RuntimeError):
        dataset.download()

    mock_download_file.assert_has_calls([
        mock.call(
            url='https://example.com/test.gz.tar',
            dirpath=tmp_path,
            filename='test.gz.tar',
            md5='52bbf03a7c50ee7152ccb9d357c2bb30'
        )
    ])


@mock.patch('pymovements.datasets.public_dataset.download_file')
def test_dataset_download_file_not_found(mock_download_file, tmp_path, dataset):
    mock_download_file.side_effect = RuntimeError()

    with pytest.raises(RuntimeError):
        dataset.download()

    mock_download_file.assert_has_calls([
        mock.call(
            url='https://example.com/test.gz.tar',
            dirpath=tmp_path,
            filename='test.gz.tar',
            md5='52bbf03a7c50ee7152ccb9d357c2bb30'
        )
    ])


@mock.patch('pymovements.datasets.public_dataset.download_file')
def test_dataset_download(mock_download_file, tmp_path, dataset):
    mock_download_file.return_value = 'path'

    dataset.download()

    mock_download_file.assert_has_calls([
        mock.call(
            url='https://example.com/test.gz.tar',
            dirpath=tmp_path,
            filename='test.gz.tar',
            md5='52bbf03a7c50ee7152ccb9d357c2bb30'
        )
    ])


@mock.patch('pymovements.datasets.public_dataset.extract_archive')
def test_dataset_extract_remove_finished_true(
        mock_extract_archive,
        tmp_path,
        dataset
):
    mock_extract_archive.return_value = 'path'

    dataset.extract(remove_finished=True)

    mock_extract_archive.assert_has_calls([
        mock.call(
            source_path=tmp_path / 'test.gz.tar',
            destination_path=tmp_path / 'raw',
            recursive=True,
            remove_finished=True,
        )
    ])


@mock.patch('pymovements.datasets.public_dataset.extract_archive')
def test_dataset_extract_remove_finished_false(
        mock_extract_archive,
        tmp_path,
        dataset
):
    mock_extract_archive.return_value = 'path'

    dataset.extract()

    mock_extract_archive.assert_has_calls([
        mock.call(
            source_path=tmp_path / 'test.gz.tar',
            destination_path=tmp_path / 'raw',
            recursive=True,
            remove_finished=False,
        )
    ])


@mock.patch('pymovements.datasets.public_dataset.PublicDataset.download')
@mock.patch('pymovements.datasets.public_dataset.PublicDataset.extract')
def test_custom_dataset_download_extract(mock_extract, mock_download, tmp_path):
    mock_extract.return_value = None
    mock_download.return_value = None

    class CustomPublicDataset(PublicDataset):
        _mirrors = ['https://example.com/']
        _resources = [
            {
                'resource': 'test.gz.tar',
                'filename': 'test.gz.tar',
                'md5': '52bbf03a7c50ee7152ccb9d357c2bb30'
            }
        ]

    CustomPublicDataset(root=tmp_path, download=True)
    CustomPublicDataset(root=tmp_path, extract=True)

    mock_download.assert_called_once()
    mock_extract.assert_called_once()
