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
"""Test all download and extract functionality of pymovements.Dataset"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from unittest import mock

import pytest

import pymovements as pm


@pytest.fixture(name='dataset_definition')
def dataset_definition_fixture():
    @dataclass
    @pm.register_dataset
    class CustomPublicDataset(pm.DatasetDefinition):
        name: str = 'CustomPublicDataset'

        mirrors: tuple[str, ...] = (
            'https://example.com/',
            'https://another_example.com/',
        )

        resources: tuple[dict[str, str], ...] = (
            {
                'resource': 'test.gz.tar',
                'filename': 'test.gz.tar',
                'md5': '52bbf03a7c50ee7152ccb9d357c2bb30',
            },
        )

    return CustomPublicDataset()


@pytest.mark.parametrize(
    ('init_path', 'expected_paths'),
    [
        pytest.param(
            '/data/set/path',
            {
                'root': Path('/data/set/path'),
                'dataset': Path('/data/set/path'),
                'downloads': Path('/data/set/path/downloads'),
            },
            id='no_paths',
        ),
        pytest.param(
            pm.DatasetPaths(root='/data/set/path'),
            {
                'root': Path('/data/set/path/'),
                'dataset': Path('/data/set/path/CustomPublicDataset'),
                'downloads': Path('/data/set/path/CustomPublicDataset/downloads'),
            },
            id='no_paths',
        ),
        pytest.param(
            pm.DatasetPaths(root='/data/set/path', dataset='.'),
            {
                'root': Path('/data/set/path/'),
                'dataset': Path('/data/set/path/'),
                'downloads': Path('/data/set/path/downloads'),
            },
            id='dataset_dot',
        ),
        pytest.param(
            pm.DatasetPaths(root='/data/set/path', dataset='dataset'),
            {
                'root': Path('/data/set/path/'),
                'dataset': Path('/data/set/path/dataset'),
                'downloads': Path('/data/set/path/dataset/downloads'),
            },
            id='explicit_dataset_dirname',
        ),
        pytest.param(
            pm.DatasetPaths(root='/data/set/path', downloads='custom_downloads'),
            {
                'root': Path('/data/set/path/'),
                'dataset': Path('/data/set/path/CustomPublicDataset'),
                'downloads': Path('/data/set/path/CustomPublicDataset/custom_downloads'),
            },
            id='explicit_download_dirname',
        ),
    ],
)
def test_paths(init_path, expected_paths, dataset_definition):
    dataset = pm.Dataset(dataset_definition, path=init_path)

    assert dataset.paths.root == expected_paths['root']
    assert dataset.paths.dataset == expected_paths['dataset']
    assert dataset.paths.downloads == expected_paths['downloads']


@mock.patch('pymovements.dataset.dataset_download.download_file')
def test_dataset_download_both_mirrors_fail(mock_download_file, tmp_path, dataset_definition):
    mock_download_file.side_effect = OSError()

    paths = pm.DatasetPaths(root=tmp_path, dataset='.')
    dataset = pm.Dataset(dataset_definition, path=paths)

    with pytest.raises(RuntimeError):
        dataset.download()

    mock_download_file.assert_has_calls([
        mock.call(
            url='https://example.com/test.gz.tar',
            dirpath=tmp_path / 'downloads',
            filename='test.gz.tar',
            md5='52bbf03a7c50ee7152ccb9d357c2bb30',
            verbose=True,
        ),
        mock.call(
            url='https://another_example.com/test.gz.tar',
            dirpath=tmp_path / 'downloads',
            filename='test.gz.tar',
            md5='52bbf03a7c50ee7152ccb9d357c2bb30',
            verbose=True,
        ),
    ])


@mock.patch('pymovements.dataset.dataset_download.download_file')
def test_dataset_download_first_mirror_fails(mock_download_file, tmp_path, dataset_definition):
    mock_download_file.side_effect = [OSError(), None]

    paths = pm.DatasetPaths(root=tmp_path, dataset='.')
    dataset = pm.Dataset(dataset_definition, path=paths)
    dataset.download(extract=False)

    mock_download_file.assert_has_calls([
        mock.call(
            url='https://example.com/test.gz.tar',
            dirpath=tmp_path / 'downloads',
            filename='test.gz.tar',
            md5='52bbf03a7c50ee7152ccb9d357c2bb30',
            verbose=True,
        ),
        mock.call(
            url='https://another_example.com/test.gz.tar',
            dirpath=tmp_path / 'downloads',
            filename='test.gz.tar',
            md5='52bbf03a7c50ee7152ccb9d357c2bb30',
            verbose=True,
        ),
    ])


@mock.patch('pymovements.dataset.dataset_download.download_file')
def test_dataset_download_file_not_found(mock_download_file, tmp_path, dataset_definition):
    mock_download_file.side_effect = RuntimeError()

    paths = pm.DatasetPaths(root=tmp_path, dataset='.')
    dataset = pm.Dataset(dataset_definition, path=paths)

    with pytest.raises(RuntimeError):
        dataset.download()

    mock_download_file.assert_has_calls([
        mock.call(
            url='https://example.com/test.gz.tar',
            dirpath=tmp_path / 'downloads',
            filename='test.gz.tar',
            md5='52bbf03a7c50ee7152ccb9d357c2bb30',
            verbose=True,
        ),
    ])


@mock.patch('pymovements.dataset.dataset_download.download_file')
def test_dataset_download_no_extract(mock_download_file, tmp_path, dataset_definition):
    mock_download_file.return_value = 'path'

    paths = pm.DatasetPaths(root=tmp_path, dataset='.')
    dataset = pm.Dataset(dataset_definition, path=paths)
    dataset.download(extract=False)

    mock_download_file.assert_has_calls([
        mock.call(
            url='https://example.com/test.gz.tar',
            dirpath=tmp_path / 'downloads',
            filename='test.gz.tar',
            md5='52bbf03a7c50ee7152ccb9d357c2bb30',
            verbose=True,
        ),
    ])


@mock.patch('pymovements.dataset.dataset_download.extract_archive')
def test_dataset_extract_remove_finished_true(
        mock_extract_archive,
        tmp_path,
        dataset_definition,
):
    mock_extract_archive.return_value = 'path'

    paths = pm.DatasetPaths(root=tmp_path, dataset='.')
    dataset = pm.Dataset(dataset_definition, path=paths)
    dataset.extract(remove_finished=True, verbose=1)

    mock_extract_archive.assert_has_calls([
        mock.call(
            source_path=tmp_path / 'downloads' / 'test.gz.tar',
            destination_path=tmp_path / 'raw',
            recursive=True,
            remove_finished=True,
            verbose=1,
        ),
    ])


@mock.patch('pymovements.dataset.dataset_download.extract_archive')
def test_dataset_extract_remove_finished_false(
        mock_extract_archive,
        tmp_path,
        dataset_definition,
):
    mock_extract_archive.return_value = 'path'

    paths = pm.DatasetPaths(root=tmp_path, dataset='.')
    dataset = pm.Dataset(dataset_definition, path=paths)
    dataset.extract()

    mock_extract_archive.assert_has_calls([
        mock.call(
            source_path=tmp_path / 'downloads' / 'test.gz.tar',
            destination_path=tmp_path / 'raw',
            recursive=True,
            remove_finished=False,
            verbose=1,
        ),
    ])


@mock.patch('pymovements.dataset.dataset_download.download_file')
@mock.patch('pymovements.dataset.dataset_download.extract_archive')
def test_dataset_download_default_extract(
        mock_extract, mock_download, tmp_path, dataset_definition,
):
    mock_extract.return_value = None
    mock_download.return_value = None

    pm.Dataset(dataset_definition, path=tmp_path).download()

    mock_download.assert_called_once()
    mock_extract.assert_called_once()


def test_dataset_download_no_mirrors_raises_exception(tmp_path):
    @dataclass
    class NoMirrorsDefinition(pm.DatasetDefinition):
        name: str = 'CustomPublicDataset'

        mirrors: tuple[str, ...] = ()

        resources: tuple[dict[str, str], ...] = (
            {
                'resource': 'test.gz.tar',
                'filename': 'test.gz.tar',
                'md5': '52bbf03a7c50ee7152ccb9d357c2bb30',
            },
        )

    with pytest.raises(AttributeError) as excinfo:
        pm.Dataset(NoMirrorsDefinition, path=tmp_path).download()

    msg, = excinfo.value.args

    expected_substrings = ['number', 'mirrors', 'zero', 'download']
    for substring in expected_substrings:
        assert substring in msg


def test_dataset_download_no_resources_raises_exception(tmp_path):
    @dataclass
    class NoResourcesDefinition(pm.DatasetDefinition):
        name: str = 'CustomPublicDataset'

        mirrors: tuple[str, ...] = (
            'https://example.com/',
            'https://another_example.com/',
        )

        resources: tuple[dict[str, str], ...] = ()

    with pytest.raises(AttributeError) as excinfo:
        pm.Dataset(NoResourcesDefinition, path=tmp_path).download()

    msg, = excinfo.value.args

    expected_substrings = ['number', 'resources', 'zero', 'download']
    for substring in expected_substrings:
        assert substring in msg


def test_public_dataset_registered_correct_attributes(tmp_path, dataset_definition):
    dataset = pm.Dataset('CustomPublicDataset', path=tmp_path)

    assert dataset.definition.mirrors == dataset_definition.mirrors
    assert dataset.definition.resources == dataset_definition.resources
    assert dataset.definition.experiment == dataset_definition.experiment
    assert dataset.definition.filename_format == dataset_definition.filename_format
    assert dataset.definition.filename_format_dtypes == dataset_definition.filename_format_dtypes
    assert dataset.definition.custom_read_kwargs == dataset_definition.custom_read_kwargs
