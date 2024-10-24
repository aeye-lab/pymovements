# Copyright (c) 2023-2024 The pymovements Project Authors
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
"""Test all download and extract functionality of pymovements.Dataset."""
from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from unittest import mock

import pytest

import pymovements as pm


@pytest.fixture(
    name='dataset_definition',
    params=[
        'CustomGazeAndPrecomputed',
        'CustomGazeOnly',
        'CustomPrecomputedOnly',
        'CustomPrecomputedOnlyNoExtract',
        'CustomPrecomputedRMOnly',
    ],
)
def dataset_definition_fixture(request):
    if request.param == 'CustomGazeAndPrecomputed':
        @dataclass
        @pm.register_dataset
        class CustomPublicDataset(pm.DatasetDefinition):
            name: str = 'CustomPublicDataset'

            has_files: dict[str, bool] = field(
                default_factory=lambda: {
                    'gaze': True,
                    'precomputed_events': True,
                    'precomputed_reading_measures': False,
                },
            )
            mirrors: dict[str, tuple[str, ...]] = field(
                default_factory=lambda: {
                    'gaze': (
                        'https://example.com/',
                        'https://another_example.com/',
                    ),
                    'precomputed_events': (
                        'https://example.com/',
                        'https://another_example.com/',
                    ),
                },
            )
            resources: dict[str, tuple[dict[str, str], ...]] = field(
                default_factory=lambda: {
                    'gaze': (
                        {
                            'resource': 'test.gz.tar',
                            'filename': 'test.gz.tar',
                            'md5': '52bbf03a7c50ee7152ccb9d357c2bb30',
                        },
                    ),
                    'precomputed_events': (
                        {
                            'resource': 'test_pc.gz.tar',
                            'filename': 'test_pc.gz.tar',
                            'md5': '52bbf03a7c50ee7152ccb9d357c2bb30',
                        },
                    ),
                },
            )
            extract: dict[str, bool] = field(
                default_factory=lambda: {
                    'gaze': True,
                    'precomputed_events': True,
                },
            )
        return CustomPublicDataset()
    if request.param == 'CustomGazeOnly':
        @dataclass
        @pm.register_dataset
        class CustomPublicDataset(pm.DatasetDefinition):
            name: str = 'CustomPublicDataset'

            has_files: dict[str, bool] = field(
                default_factory=lambda: {
                    'gaze': True,
                    'precomputed_events': False,
                    'precomputed_reading_measures': False,
                },
            )
            mirrors: dict[str, [tuple[str, ...]]] = field(
                default_factory=lambda: {
                    'gaze': (
                        'https://example.com/',
                        'https://another_example.com/',
                    ),
                },
            )

            resources: dict[str, tuple[dict[str, str], ...]] = field(
                default_factory=lambda: {
                    'gaze': (
                        {
                            'resource': 'test.gz.tar',
                            'filename': 'test.gz.tar',
                            'md5': '52bbf03a7c50ee7152ccb9d357c2bb30',
                        },
                    ),
                },
            )
            extract: dict[str, bool] = field(default_factory=lambda: {'gaze': True})

        return CustomPublicDataset()
    if request.param == 'CustomPrecomputedOnly':
        @dataclass
        @pm.register_dataset
        class CustomPublicDataset(pm.DatasetDefinition):
            name: str = 'CustomPublicDataset'

            has_files: dict[str, bool] = field(
                default_factory=lambda: {
                    'gaze': False,
                    'precomputed_events': True,
                    'precomputed_reading_measures': False,
                },
            )
            extract: dict[str, bool] = field(default_factory=lambda: {'precomputed_events': True})
            mirrors: dict[str, [tuple[str, ...]]] = field(
                default_factory=lambda: {
                    'precomputed_events': (
                        'https://example.com/',
                        'https://another_example.com/',
                    ),
                },
            )

            resources: dict[str, tuple[dict[str, str], ...]] = field(
                default_factory=lambda: {
                    'precomputed_events': (
                        {
                            'resource': 'test_pc.gz.tar',
                            'filename': 'test_pc.gz.tar',
                            'md5': '52bbf03a7c50ee7152ccb9d357c2bb30',
                        },
                    ),
                },
            )

        return CustomPublicDataset()
    if request.param == 'CustomPrecomputedOnlyNoExtract':
        @dataclass
        @pm.register_dataset
        class CustomPublicDataset(pm.DatasetDefinition):
            name: str = 'CustomPublicDataset'

            has_files: dict[str, bool] = field(
                default_factory=lambda: {
                    'gaze': False,
                    'precomputed_events': True,
                    'precomputed_reading_measures': False,
                },
            )
            extract: dict[str, bool] = field(default_factory=lambda: {'precomputed_events': False})
            mirrors: dict[str, [tuple[str, ...]]] = field(
                default_factory=lambda: {
                    'precomputed_events': (
                        'https://example.com/',
                        'https://another_example.com/',
                    ),
                },
            )

            resources: dict[str, tuple[dict[str, str], ...]] = field(
                default_factory=lambda: {
                    'precomputed_events': (
                        {
                            'resource': 'test_pc.gz.tar',
                            'filename': 'test_pc.gz.tar',
                            'md5': '52bbf03a7c50ee7152ccb9d357c2bb30',
                        },
                    ),
                },
            )

        return CustomPublicDataset()
    if request.param == 'CustomPrecomputedRMOnly':
        @dataclass
        @pm.register_dataset
        class CustomPublicDataset(pm.DatasetDefinition):
            name: str = 'CustomPublicDataset'

            has_files: dict[str, bool] = field(
                default_factory=lambda: {
                    'gaze': False,
                    'precomputed_events': False,
                    'precomputed_reading_measures': True,
                },
            )
            extract: dict[str, bool] = field(
                default_factory=lambda: {
                    'precomputed_reading_measures': True,
                },
            )
            mirrors: dict[str, [tuple[str, ...]]] = field(
                default_factory=lambda: {
                    'precomputed_reading_measures': (
                        'https://example.com/',
                        'https://another_example.com/',
                    ),
                },
            )

            resources: dict[str, tuple[dict[str, str], ...]] = field(
                default_factory=lambda: {
                    'precomputed_reading_measures': (
                        {
                            'resource': 'test_rm.gz.tar',
                            'filename': 'test_rm.gz.tar',
                            'md5': '52bbf03a7c50ee7152ccb9d357c2bb30',
                        },
                    ),
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
@pytest.mark.parametrize('dataset_definition', ['CustomGazeOnly'], indirect=['dataset_definition'])
def test_dataset_download_both_mirrors_fail_gaze_only(
        mock_download_file,
        tmp_path,
        dataset_definition,
):
    paths = pm.DatasetPaths(root=tmp_path, dataset='.')
    dataset = pm.Dataset(dataset_definition, path=paths)

    mock_download_file.side_effect = OSError

    with pytest.raises(
        RuntimeError,
        match='downloading resource test.gz.tar failed for all mirrors',
    ):
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
@pytest.mark.parametrize(
    'dataset_definition',
    ['CustomPrecomputedOnly'],
    indirect=['dataset_definition'],
)
def test_dataset_download_both_precomputed_mirrors_fail(
        mock_download_file,
        tmp_path,
        dataset_definition,
):
    mock_download_file.side_effect = OSError()

    paths = pm.DatasetPaths(root=tmp_path, dataset='.')
    dataset = pm.Dataset(dataset_definition, path=paths)

    with pytest.raises(
        RuntimeError,
        match='downloading resource test_pc.gz.tar failed for all mirrors',
    ):
        dataset.download()

    mock_download_file.assert_has_calls([
        mock.call(
            url='https://example.com/test_pc.gz.tar',
            dirpath=tmp_path / 'downloads',
            filename='test_pc.gz.tar',
            md5='52bbf03a7c50ee7152ccb9d357c2bb30',
            verbose=True,
        ),
        mock.call(
            url='https://another_example.com/test_pc.gz.tar',
            dirpath=tmp_path / 'downloads',
            filename='test_pc.gz.tar',
            md5='52bbf03a7c50ee7152ccb9d357c2bb30',
            verbose=True,
        ),
    ])


@mock.patch('pymovements.dataset.dataset_download.download_file')
@pytest.mark.parametrize(
    'dataset_definition',
    ['CustomPrecomputedRMOnly'],
    indirect=['dataset_definition'],
)
def test_dataset_download_both_precomputed_mirrors_fail_rm(
        mock_download_file,
        tmp_path,
        dataset_definition,
):
    mock_download_file.side_effect = OSError()

    paths = pm.DatasetPaths(root=tmp_path, dataset='.')
    dataset = pm.Dataset(dataset_definition, path=paths)

    with pytest.raises(
        RuntimeError,
        match='downloading resource test_rm.gz.tar failed for all mirrors',
    ):
        dataset.download()

    mock_download_file.assert_has_calls([
        mock.call(
            url='https://example.com/test_rm.gz.tar',
            dirpath=tmp_path / 'downloads',
            filename='test_rm.gz.tar',
            md5='52bbf03a7c50ee7152ccb9d357c2bb30',
            verbose=True,
        ),
        mock.call(
            url='https://another_example.com/test_rm.gz.tar',
            dirpath=tmp_path / 'downloads',
            filename='test_rm.gz.tar',
            md5='52bbf03a7c50ee7152ccb9d357c2bb30',
            verbose=True,
        ),
    ])


@mock.patch('pymovements.dataset.dataset_download.download_file')
@pytest.mark.parametrize(
    'dataset_definition',
    ['CustomGazeAndPrecomputed'],
    indirect=['dataset_definition'],
)
def test_dataset_download_both_mirrors_fail_precomputed_and_gaze(
        mock_download_file,
        tmp_path,
        dataset_definition,
):
    mock_download_file.side_effect = OSError()

    paths = pm.DatasetPaths(root=tmp_path, dataset='.')
    dataset = pm.Dataset(dataset_definition, path=paths)

    with pytest.raises(
        RuntimeError,
        match='downloading resource test.gz.tar failed for all mirrors',
    ):
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
@pytest.mark.parametrize('dataset_definition', ['CustomGazeOnly'], indirect=['dataset_definition'])
def test_dataset_download_first_mirror_gaze_fails(mock_download_file, tmp_path, dataset_definition):
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
@pytest.mark.parametrize(
    'dataset_definition',
    ['CustomPrecomputedOnly'], indirect=['dataset_definition'],
)
def test_dataset_download_first_mirror_precomputed_fails(
        mock_download_file, tmp_path, dataset_definition,
):
    mock_download_file.side_effect = [OSError(), None]

    paths = pm.DatasetPaths(root=tmp_path, dataset='.')
    dataset = pm.Dataset(dataset_definition, path=paths)
    dataset.download(extract=False)
    mock_download_file.assert_has_calls([
        mock.call(
            url='https://example.com/test_pc.gz.tar',
            dirpath=tmp_path / 'downloads',
            filename='test_pc.gz.tar',
            md5='52bbf03a7c50ee7152ccb9d357c2bb30',
            verbose=True,
        ),
        mock.call(
            url='https://another_example.com/test_pc.gz.tar',
            dirpath=tmp_path / 'downloads',
            filename='test_pc.gz.tar',
            md5='52bbf03a7c50ee7152ccb9d357c2bb30',
            verbose=True,
        ),
    ])


@mock.patch('pymovements.dataset.dataset_download.download_file')
@pytest.mark.parametrize(
    'dataset_definition',
    ['CustomPrecomputedRMOnly'], indirect=['dataset_definition'],
)
def test_dataset_download_first_mirror_precomputed_fails_rm(
        mock_download_file, tmp_path, dataset_definition,
):
    mock_download_file.side_effect = [OSError(), None]

    paths = pm.DatasetPaths(root=tmp_path, dataset='.')
    dataset = pm.Dataset(dataset_definition, path=paths)
    dataset.download(extract=False)
    mock_download_file.assert_has_calls([
        mock.call(
            url='https://example.com/test_rm.gz.tar',
            dirpath=tmp_path / 'downloads',
            filename='test_rm.gz.tar',
            md5='52bbf03a7c50ee7152ccb9d357c2bb30',
            verbose=True,
        ),
        mock.call(
            url='https://another_example.com/test_rm.gz.tar',
            dirpath=tmp_path / 'downloads',
            filename='test_rm.gz.tar',
            md5='52bbf03a7c50ee7152ccb9d357c2bb30',
            verbose=True,
        ),
    ])


@mock.patch('pymovements.dataset.dataset_download.download_file')
@pytest.mark.parametrize(
    'dataset_definition',
    ['CustomGazeAndPrecomputed'], indirect=['dataset_definition'],
)
def test_dataset_download_first_mirror_fails(mock_download_file, tmp_path, dataset_definition):
    mock_download_file.side_effect = [OSError(), None, OSError(), None]

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
        mock.call(
            url='https://example.com/test_pc.gz.tar',
            dirpath=tmp_path / 'downloads',
            filename='test_pc.gz.tar',
            md5='52bbf03a7c50ee7152ccb9d357c2bb30',
            verbose=True,
        ),
        mock.call(
            url='https://another_example.com/test_pc.gz.tar',
            dirpath=tmp_path / 'downloads',
            filename='test_pc.gz.tar',
            md5='52bbf03a7c50ee7152ccb9d357c2bb30',
            verbose=True,
        ),
    ])


@mock.patch('pymovements.dataset.dataset_download.download_file')
@pytest.mark.parametrize(
    'dataset_definition',
    ['CustomGazeOnly', 'CustomGazeAndPrecomputed'],
    indirect=['dataset_definition'],
)
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
@pytest.mark.parametrize(
    'dataset_definition',
    ['CustomPrecomputedOnly'],
    indirect=['dataset_definition'],
)
def test_dataset_download_file_precomputed_not_found(
        mock_download_file, tmp_path, dataset_definition,
):
    mock_download_file.side_effect = RuntimeError()

    paths = pm.DatasetPaths(root=tmp_path, dataset='.')
    dataset = pm.Dataset(dataset_definition, path=paths)

    with pytest.raises(RuntimeError):
        dataset.download()

    mock_download_file.assert_has_calls([
        mock.call(
            url='https://example.com/test_pc.gz.tar',
            dirpath=tmp_path / 'downloads',
            filename='test_pc.gz.tar',
            md5='52bbf03a7c50ee7152ccb9d357c2bb30',
            verbose=True,
        ),
    ])


@mock.patch('pymovements.dataset.dataset_download.download_file')
@pytest.mark.parametrize(
    'dataset_definition',
    ['CustomGazeOnly', 'CustomGazeAndPrecomputed'],
    indirect=['dataset_definition'],
)
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


@mock.patch('pymovements.dataset.dataset_download.download_file')
@pytest.mark.parametrize(
    'dataset_definition',
    ['CustomPrecomputedOnly'], indirect=['dataset_definition'],
)
def test_dataset_download_precomputed_no_extract(mock_download_file, tmp_path, dataset_definition):
    mock_download_file.return_value = 'path'

    paths = pm.DatasetPaths(root=tmp_path, dataset='.')
    dataset = pm.Dataset(dataset_definition, path=paths)
    dataset.download(extract=False)

    mock_download_file.assert_has_calls([
        mock.call(
            url='https://example.com/test_pc.gz.tar',
            dirpath=tmp_path / 'downloads',
            filename='test_pc.gz.tar',
            md5='52bbf03a7c50ee7152ccb9d357c2bb30',
            verbose=True,
        ),
    ])


@mock.patch('pymovements.dataset.dataset_download.download_file')
@pytest.mark.parametrize(
    'dataset_definition',
    ['CustomPrecomputedRMOnly'], indirect=['dataset_definition'],
)
def test_dataset_download_precomputed_no_extract_rm(
        mock_download_file, tmp_path, dataset_definition,
):
    mock_download_file.return_value = 'path'

    paths = pm.DatasetPaths(root=tmp_path, dataset='.')
    dataset = pm.Dataset(dataset_definition, path=paths)
    dataset.download(extract=False)

    mock_download_file.assert_has_calls([
        mock.call(
            url='https://example.com/test_rm.gz.tar',
            dirpath=tmp_path / 'downloads',
            filename='test_rm.gz.tar',
            md5='52bbf03a7c50ee7152ccb9d357c2bb30',
            verbose=True,
        ),
    ])


@mock.patch('pymovements.dataset.dataset_download.extract_archive')
@pytest.mark.parametrize('dataset_definition', ['CustomGazeOnly'], indirect=['dataset_definition'])
def test_dataset_extract_remove_finished_true_gaze(
        mock_extract_archive,
        tmp_path,
        dataset_definition,
):
    mock_extract_archive.return_value = 'path'

    paths = pm.DatasetPaths(root=tmp_path, dataset='.')
    dataset = pm.Dataset(dataset_definition, path=paths)
    dataset.extract(remove_finished=True, remove_top_level=False, verbose=1)

    mock_extract_archive.assert_has_calls([
        mock.call(
            source_path=tmp_path / 'downloads' / 'test.gz.tar',
            destination_path=tmp_path / 'raw',
            recursive=True,
            remove_finished=True,
            remove_top_level=False,
            verbose=1,
        ),
    ])


@mock.patch('pymovements.dataset.dataset_download.extract_archive')
@pytest.mark.parametrize(
    'dataset_definition',
    ['CustomPrecomputedRMOnly'],
    indirect=['dataset_definition'],
)
def test_dataset_extract_rm(
        mock_extract_archive,
        tmp_path,
        dataset_definition,
):
    mock_extract_archive.return_value = 'path'

    paths = pm.DatasetPaths(root=tmp_path, dataset='.')
    dataset = pm.Dataset(dataset_definition, path=paths)
    dataset.extract(verbose=1)

    mock_extract_archive.assert_has_calls([
        mock.call(
            source_path=tmp_path / 'downloads' / 'test_rm.gz.tar',
            destination_path=tmp_path / 'precomputed_reading_measures',
            recursive=True,
            remove_finished=False,
            remove_top_level=True,
            verbose=1,
        ),
    ])


@mock.patch('pymovements.dataset.dataset_download.extract_archive')
@pytest.mark.parametrize(
    'dataset_definition',
    ['CustomGazeAndPrecomputed'], indirect=['dataset_definition'],
)
def test_dataset_extract_remove_finished_true_both(
        mock_extract_archive,
        tmp_path,
        dataset_definition,
):
    mock_extract_archive.return_value = 'path'

    paths = pm.DatasetPaths(root=tmp_path, dataset='.')
    dataset = pm.Dataset(dataset_definition, path=paths)
    dataset.extract(remove_finished=True, remove_top_level=False, verbose=1)

    mock_extract_archive.assert_has_calls([
        mock.call(
            source_path=tmp_path / 'downloads' / 'test.gz.tar',
            destination_path=tmp_path / 'raw',
            recursive=True,
            remove_finished=True,
            remove_top_level=False,
            verbose=1,
        ),
        mock.call(
            source_path=tmp_path / 'downloads' / 'test_pc.gz.tar',
            destination_path=tmp_path / 'precomputed_events',
            recursive=True,
            remove_finished=True,
            remove_top_level=False,
            verbose=1,
        ),
    ])


@mock.patch('pymovements.dataset.dataset_download.extract_archive')
@pytest.mark.parametrize(
    'dataset_definition',
    ['CustomPrecomputedOnly'], indirect=['dataset_definition'],
)
def test_dataset_extract_remove_finished_true_precomputed(
        mock_extract_archive,
        tmp_path,
        dataset_definition,
):
    mock_extract_archive.return_value = 'path'

    paths = pm.DatasetPaths(root=tmp_path, dataset='.')
    dataset = pm.Dataset(dataset_definition, path=paths)
    dataset.extract(remove_finished=True, remove_top_level=False, verbose=1)

    mock_extract_archive.assert_has_calls([
        mock.call(
            source_path=tmp_path / 'downloads' / 'test_pc.gz.tar',
            destination_path=tmp_path / 'precomputed_events',
            recursive=True,
            remove_finished=True,
            remove_top_level=False,
            verbose=1,
        ),
    ])


@mock.patch('pymovements.dataset.dataset_download.extract_archive')
@pytest.mark.parametrize(
    'dataset_definition',
    ['CustomGazeAndPrecomputed'], indirect=['dataset_definition'],
)
def test_dataset_extract_remove_finished_false_both(
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
            remove_top_level=True,
            verbose=1,
        ),
        mock.call(
            source_path=tmp_path / 'downloads' / 'test_pc.gz.tar',
            destination_path=tmp_path / 'precomputed_events',
            recursive=True,
            remove_finished=False,
            remove_top_level=True,
            verbose=1,
        ),
    ])


@mock.patch('pymovements.dataset.dataset_download.extract_archive')
@pytest.mark.parametrize('dataset_definition', ['CustomGazeOnly'], indirect=['dataset_definition'])
def test_dataset_extract_remove_finished_false_gaze(
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
            remove_top_level=True,
            verbose=1,
        ),
    ])


@mock.patch('pymovements.dataset.dataset_download.extract_archive')
@pytest.mark.parametrize(
    'dataset_definition',
    ['CustomPrecomputedOnly'], indirect=['dataset_definition'],
)
def test_dataset_extract_remove_finished_false_precomputed(
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
            source_path=tmp_path / 'downloads' / 'test_pc.gz.tar',
            destination_path=tmp_path / 'precomputed_events',
            recursive=True,
            remove_finished=False,
            remove_top_level=True,
            verbose=1,
        ),
    ])


@mock.patch('pymovements.dataset.dataset_download.download_file')
@mock.patch('pymovements.dataset.dataset_download.extract_archive')
@pytest.mark.parametrize(
    'dataset_definition',
    ['CustomGazeAndPrecomputed'], indirect=['dataset_definition'],
)
def test_dataset_download_default_extract_both(
        mock_extract, mock_download, tmp_path, dataset_definition,
):
    mock_extract.return_value = None
    mock_download.return_value = None

    pm.Dataset(dataset_definition, path=tmp_path).download()


@mock.patch('pymovements.dataset.dataset_download.download_file')
@mock.patch('pymovements.dataset.dataset_download.extract_archive')
@pytest.mark.parametrize('dataset_definition', ['CustomGazeOnly'], indirect=['dataset_definition'])
def test_dataset_download_default_extract_gaze(
        mock_extract, mock_download, tmp_path, dataset_definition,
):
    mock_extract.return_value = None
    mock_download.return_value = None

    pm.Dataset(dataset_definition, path=tmp_path).download()

    mock_download.assert_called_once()
    mock_extract.assert_called_once()


@mock.patch('pymovements.dataset.dataset_download.download_file')
@mock.patch('pymovements.dataset.dataset_download.extract_archive')
@pytest.mark.parametrize(
    'dataset_definition',
    ['CustomPrecomputedOnly'], indirect=['dataset_definition'],
)
def test_dataset_download_default_extract_precomputed(
        mock_extract, mock_download, tmp_path, dataset_definition,
):
    mock_extract.return_value = None
    mock_download.return_value = None

    pm.Dataset(dataset_definition, path=tmp_path).download()

    mock_download.assert_called_once()
    mock_extract.assert_called_once()


def test_dataset_download_no_mirrors_raises_exception(tmp_path):
    @dataclass
    class NoGazeMirrorsDefinition(pm.DatasetDefinition):
        name: str = 'CustomPublicDataset'

        has_files: dict[str, bool] = field(
            default_factory=lambda: {
                'gaze': True,
                'precomputed_events': False,
                'precomputed_reading_measures': False,
            },
        )
        mirrors: dict[str, tuple[str, ...]] = field(default_factory=lambda: {'gaze': ()})

        resources: dict[str, tuple[dict[str, str], ...]] = field(
            default_factory=lambda: {
                'gaze': (
                    {
                        'resource': 'test.gz.tar',
                        'filename': 'test.gz.tar',
                        'md5': '52bbf03a7c50ee7152ccb9d357c2bb30',
                    },
                ),
            },
        )

    with pytest.raises(AttributeError) as excinfo:
        pm.Dataset(NoGazeMirrorsDefinition, path=tmp_path).download()

    msg, = excinfo.value.args

    expected_substrings = ['number', 'mirrors', 'zero', 'download']
    for substring in expected_substrings:
        assert substring in msg


def test_dataset_download_no_mirrors_precomputed_raises_exception(tmp_path):
    @dataclass
    class NoPrecomputedMirrorsDefinition(pm.DatasetDefinition):
        name: str = 'CustomPublicDataset'

        has_files: dict[str, bool] = field(
            default_factory=lambda: {
                'gaze': False,
                'precomputed_events': True,
                'precomputed_reading_measures': False,
            },
        )
        mirrors: dict[str, tuple[str, ...]] = field(
            default_factory=lambda: {
                'precomputed_events': (),
            },
        )

        resources: dict[str, tuple[dict[str, str], ...]] = field(
            default_factory=lambda: {
                'precomputed_events': (
                    {
                        'resource': 'test_pc.gz.tar',
                        'filename': 'test_pc.gz.tar',
                        'md5': '52bbf03a7c50ee7152ccb9d357c2bb30',
                    },
                ),
            },
        )

    with pytest.raises(AttributeError) as excinfo:
        pm.Dataset(NoPrecomputedMirrorsDefinition, path=tmp_path).download()

    msg, = excinfo.value.args

    expected_substrings = ['number', 'mirrors', 'zero', 'download']
    for substring in expected_substrings:
        assert substring in msg


def test_dataset_download_no_mirrors_precomputed_rm_raises_exception(tmp_path):
    @dataclass
    class NoPrecomputedMirrorsDefinition(pm.DatasetDefinition):
        name: str = 'CustomPublicDataset'

        has_files: dict[str, bool] = field(
            default_factory=lambda: {
                'gaze': False,
                'precomputed_events': False,
                'precomputed_reading_measures': True,
            },
        )
        mirrors: dict[str, tuple[str, ...]] = field(
            default_factory=lambda: {
                'precomputed_reading_measures': (),
            },
        )

        resources: dict[str, tuple[dict[str, str], ...]] = field(
            default_factory=lambda: {
                'precomputed_reading_measures': (
                    {
                        'resource': 'test_rm.gz.tar',
                        'filename': 'test_rm.gz.tar',
                        'md5': '52bbf03a7c50ee7152ccb9d357c2bb30',
                    },
                ),
            },
        )

    with pytest.raises(AttributeError) as excinfo:
        pm.Dataset(NoPrecomputedMirrorsDefinition, path=tmp_path).download()

    msg, = excinfo.value.args

    expected_substrings = ['number', 'mirrors', 'zero', 'download']
    for substring in expected_substrings:
        assert substring in msg


def test_dataset_download_no_resources_raises_exception(tmp_path):
    @dataclass
    class NoGazeResourcesDefinition(pm.DatasetDefinition):
        name: str = 'CustomPublicDataset'

        has_files: dict[str, bool] = field(
            default_factory=lambda: {
                'gaze': True,
                'precomputed_events': False,
                'precomputed_reading_measures': False,
            },
        )
        mirrors: dict[str, tuple[str, ...]] = field(
            default_factory=lambda: {
                'gaze': (
                    'https://example.com/',
                    'https://another_example.com/',
                ),
            },
        )

        resources: dict[str, tuple[dict[str, str], ...]] = field(
            default_factory=lambda: {
                'gaze': (),
            },
        )

    with pytest.raises(AttributeError) as excinfo:
        pm.Dataset(NoGazeResourcesDefinition, path=tmp_path).download()

    msg, = excinfo.value.args

    expected_substrings = ['number', 'resources', 'zero', 'download']
    for substring in expected_substrings:
        assert substring in msg


def test_dataset_download_no_precomputed_event_resources_raises_exception(tmp_path):
    @dataclass
    class NoPrecomputedResourcesDefinition(pm.DatasetDefinition):
        name: str = 'CustomPublicDataset'

        has_files: dict[str, bool] = field(
            default_factory=lambda: {
                'gaze': False,
                'precomputed_events': True,
                'precomputed_reading_measures': False,
            },
        )
        mirrors: dict[str, tuple[str, ...]] = field(
            default_factory=lambda: {
                'precomputed_events': (
                    'https://example.com/',
                    'https://another_example.com/',
                ),
            },
        )

        resources: dict[str, tuple[dict[str, str], ...]] = field(
            default_factory=lambda: {
                'precomputed_events': (),
            },
        )

    with pytest.raises(AttributeError) as excinfo:
        pm.Dataset(NoPrecomputedResourcesDefinition, path=tmp_path).download()

    msg, = excinfo.value.args

    expected_substrings = ['number', 'precomputed_event_resources', 'zero', 'download']
    for substring in expected_substrings:
        assert substring in msg


def test_public_dataset_registered_correct_attributes(tmp_path, dataset_definition):
    dataset = pm.Dataset(dataset_definition, path=tmp_path)

    assert dataset.definition.mirrors == dataset_definition.mirrors
    assert dataset.definition.resources == dataset_definition.resources
    assert dataset.definition.experiment == dataset_definition.experiment
    assert dataset.definition.filename_format == dataset_definition.filename_format
    assert dataset.definition.filename_format_schema_overrides == dataset_definition.filename_format_schema_overrides  # noqa: E501
    assert dataset.definition.has_files == dataset_definition.has_files


def test_extract_dataset_precomputed_move_single_file():
    @dataclass
    @pm.register_dataset
    class PrecomputedResourcesDefinition(pm.DatasetDefinition):
        name: str = 'CustomPublicDataset'

        has_files: dict[str, bool] = field(
            default_factory=lambda: {
                'gaze': False,
                'precomputed_events': True,
                'precomputed_reading_measures': False,
            },
        )
        mirrors: dict[str, tuple[str, ...]] = field(
            default_factory=lambda: {
                'precompued_events': (
                    'https://example.com/',
                    'https://another_example.com/',
                ),
            },
        )

        resources: dict[str, tuple[dict[str, str], ...]] = field(
            default_factory=lambda: {
                'precomputed_events': (
                    {
                        'resource': 'tests/files/',
                        'filename': '18sat_fixfinal.csv',
                        'md5': '52bbf03a7c50ee7152ccb9d357c2bb30',
                    },
                ),
            },
        )
        extract: dict[str, bool] = field(default_factory=lambda: {'precomputed_events': False})

    pm.dataset.dataset_download.extract_dataset(
        PrecomputedResourcesDefinition(),
        pm.DatasetPaths(root='tests/files/', downloads='.'),
    )


def test_extract_dataset_precomputed_rm_move_single_file():
    @dataclass
    @pm.register_dataset
    class PrecomputedResourcesDefinition(pm.DatasetDefinition):
        name: str = 'CustomPublicDataset'

        has_files: dict[str, bool] = field(
            default_factory=lambda: {
                'gaze': False,
                'precomputed_events': False,
                'precomputed_reading_measures': True,
            },
        )
        mirrors: dict[str, tuple[str, ...]] = field(
            default_factory=lambda: {
                'precomputed_reading_measures': (
                    'https://example.com/',
                    'https://another_example.com/',
                ),
            },
        )

        resources: dict[str, tuple[dict[str, str], ...]] = field(
            default_factory=lambda: {
                'precomputed_reading_measures': (
                    {
                        'resource': 'tests/files/',
                        'filename': 'copco_rm_dummy.csv',
                        'md5': '52bbf03a7c50ee7152ccb9d357c2bb30',
                    },
                ),
            },
        )
        extract: dict[str, bool] = field(
            default_factory=lambda: {
                'precomputed_reading_measures': False,
            },
        )

    pm.dataset.dataset_download.extract_dataset(
        PrecomputedResourcesDefinition(),
        pm.DatasetPaths(root='tests/files/', downloads='.', precomputed_reading_measures='.'),
    )


def test_dataset_download_no_precomputed_rm_resources_raises_exception(tmp_path):
    @dataclass
    class NoPrecomputedResourcesDefinition(pm.DatasetDefinition):
        name: str = 'CustomPublicDataset'

        has_files: dict[str, bool] = field(
            default_factory=lambda: {
                'gaze': False,
                'precomputed_events': False,
                'precomputed_reading_measures': True,
            },
        )
        mirrors: dict[str, tuple[str, ...]] = field(
            default_factory=lambda: {
                'precomputed_reading_measures': (
                    'https://example.com/',
                    'https://another_example.com/',
                ),
            },
        )

        resources: dict[str, tuple[dict[str, str], ...]] = field(
            default_factory=lambda: {
                'precomputed_reading_measures': (),
            },
        )

    with pytest.raises(AttributeError) as excinfo:
        pm.Dataset(NoPrecomputedResourcesDefinition, path=tmp_path).download()

    msg, = excinfo.value.args

    expected_substrings = ['number', 'precomputed_reading_measures resources', 'zero', 'download']
    for substring in expected_substrings:
        assert substring in msg
