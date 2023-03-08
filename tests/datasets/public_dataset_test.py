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


def test_dataset_download_fails(tmp_path):
    class CustomPublicDataset(PublicDataset):
        _mirrors = ['https://www.example.com/', 'https://www.example.com/']
        _resources = [
            {
                'resource': 'v0.4.0.tar.gz',
                'filename': 'pymovements-0.4.0.tar.gz',
                'md5': '52bbf03a7c50ee7152ccb9d357c2bb30',
            },
        ]

    with mock.patch(
            'pymovements.utils.downloads._download_url',
            side_effect=OSError(),
    ):
        with pytest.raises(RuntimeError):
            CustomPublicDataset(root=tmp_path, download=True)


def test_dataset_download_file_not_found(tmp_path):
    class CustomPublicDataset(PublicDataset):
        _mirrors = ['https://github.com/aeye-lab/pymovements/archive/refs/tags/']
        _resources = [
            {
                'resource': 'v0.4.0.tar.gz',
                'filename': 'pymovements-0.4.0.tar.gz',
                'md5': '52bbf03a7c50ee7152ccb9d357c2bb30',
            },
        ]

    with mock.patch(
            'pymovements.datasets.public_dataset.PublicDataset.download',
            side_effect=RuntimeError(),
    ):
        with pytest.raises(RuntimeError):
            CustomPublicDataset(root=tmp_path, download=True)


def test_dataset_download(tmp_path):

    class CustomPublicDataset(PublicDataset):
        _mirrors = [
            'https://github.com/aeye-lab/pymovements/archive/refs/tags/',
            'https://github.com/aeye-lab/pymovements/archive/refs/tags/']
        _resources = [
            {
                'resource': 'v0.4.0.tar.gz',
                'filename': 'pymovements-0.4.0.tar.gz',
                'md5': '52bbf03a7c50ee7152ccb9d357c2bb30',
            },
        ]

    CustomPublicDataset(root=tmp_path, download=True)


@pytest.fixture(
    name='single_archive',
    params=[
        ('gz', 'tar'),
    ],
    ids=[
        'gz_compressed_tar_archive',
    ],
)
def fixture_archive(tmp_path):
    rootpath = tmp_path
    compression, extension = 'tar', 'gz'

    # write tmp filepath
    filepath = rootpath / 'test.file'
    filepath.write_text('test')

    archive_path = rootpath / f'test.{extension}.{compression}'

    with tarfile.TarFile.open(archive_path, f'w:{compression}') as fp:
        fp.add(filepath)

    yield archive_path


def test_dataset_extract(tmp_path, single_archive):

    class CustomPublicDataset(PublicDataset):
        _mirrors = []
        _resources = [
            {
                'resource': 'test.tar.gz',
                'filename': 'test.tar.gz',
                'md5': '52bbf03a7c50ee7152ccb9d357c2bb30',
            },
        ]

    CustomPublicDataset(root=tmp_path, extract=True, dataset_dirname='', downloads_dirname='')


