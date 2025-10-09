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
"""Test all download and extract functionality of pymovements.Dataset."""
from __future__ import annotations

import shutil
from pathlib import Path
from unittest import mock

import pytest

from pymovements import Dataset
from pymovements import DatasetDefinition
from pymovements import DatasetPaths


@pytest.fixture(
    name='dataset_definition',
    params=[
        'CustomGazeAndPrecomputed',
        'CustomGazeAndPrecomputedNoMirror',
        'CustomGazeOnly',
        'CustomGazeOnlyNoMirror',
        'CustomPrecomputedOnly',
        'CustomPrecomputedOnlyNoMirror',
        'CustomPrecomputedOnlyNoExtract',
        'CustomPrecomputedOnlyNoExtractNoMirror',
        'CustomPrecomputedRMOnly',
        'CustomPrecomputedRMOnlyNoMirror',
    ],
)
def dataset_definition_fixture(request):
    if request.param == 'CustomGazeAndPrecomputed':
        return DatasetDefinition(
            name='CustomPublicDataset',
            mirrors={
                'gaze': (
                    'https://example.com/',
                    'https://another_example.com/',
                ),
                'precomputed_events': (
                    'https://example.com/',
                    'https://another_example.com/',
                ),
            },
            resources=[
                {
                    'content': 'gaze',
                    'url': 'test.gz.tar',
                    'filename': 'test.gz.tar',
                    'md5': '52bbf03a7c50ee7152ccb9d357c2bb30',
                },
                {
                    'content': 'precomputed_events',
                    'url': 'test_pc.gz.tar',
                    'filename': 'test_pc.gz.tar',
                    'md5': '52bbf03a7c50ee7152ccb9d357c2bb30',
                },
            ],
        )

    if request.param == 'CustomGazeAndPrecomputedNoMirror':
        return DatasetDefinition(
            name='CustomPublicDataset',
            resources=[
                {
                    'content': 'gaze',
                    'url': 'https://example.com/test.gz.tar',
                    'filename': 'test.gz.tar',
                    'md5': '52bbf03a7c50ee7152ccb9d357c2bb30',
                },
                {
                    'content': 'precomputed_events',
                    'url': 'https://example.com/test_pc.gz.tar',
                    'filename': 'test_pc.gz.tar',
                    'md5': '52bbf03a7c50ee7152ccb9d357c2bb30',
                },
            ],
        )

    if request.param == 'CustomGazeOnly':
        return DatasetDefinition(
            name='CustomPublicDataset',
            mirrors={
                'gaze': (
                    'https://example.com/',
                    'https://another_example.com/',
                ),
            },
            resources=[
                {
                    'content': 'gaze',
                    'url': 'test.gz.tar',
                    'filename': 'test.gz.tar',
                    'md5': '52bbf03a7c50ee7152ccb9d357c2bb30',
                },
            ],
        )

    if request.param == 'CustomGazeOnlyNoMirror':
        return DatasetDefinition(
            name='CustomPublicDataset',
            resources=[
                {
                    'content': 'gaze',
                    'url': 'https://example.com/test.gz.tar',
                    'filename': 'test.gz.tar',
                    'md5': '52bbf03a7c50ee7152ccb9d357c2bb30',
                },
            ],
        )

    if request.param == 'CustomPrecomputedOnly':
        return DatasetDefinition(
            name='CustomPublicDataset',
            mirrors={
                'precomputed_events': (
                    'https://example.com/',
                    'https://another_example.com/',
                ),
            },
            resources=[
                {
                    'content': 'precomputed_events',
                    'url': 'test_pc.gz.tar',
                    'filename': 'test_pc.gz.tar',
                    'md5': '52bbf03a7c50ee7152ccb9d357c2bb30',
                },
            ],
        )

    if request.param == 'CustomPrecomputedOnlyNoMirror':
        return DatasetDefinition(
            name='CustomPublicDataset',
            resources=[
                {
                    'content': 'precomputed_events',
                    'url': 'https://example.com/test_pc.gz.tar',
                    'filename': 'test_pc.gz.tar',
                    'md5': '52bbf03a7c50ee7152ccb9d357c2bb30',
                },
            ],
        )

    if request.param == 'CustomPrecomputedOnlyNoExtract':
        return DatasetDefinition(
            name='CustomPublicDataset',
            mirrors={
                'precomputed_events': (
                    'https://example.com/',
                    'https://another_example.com/',
                ),
            },
            resources=[
                {
                    'content': 'precomputed_events',
                    'url': 'test_pc.gz.tar',
                    'filename': 'test_pc.gz.tar',
                    'md5': '52bbf03a7c50ee7152ccb9d357c2bb30',
                },
            ],
        )

    if request.param == 'CustomPrecomputedOnlyNoExtractNoMirror':
        return DatasetDefinition(
            name='CustomPublicDataset',
            resources=[
                {
                    'content': 'precomputed_events',
                    'url': 'https://example.com/test_pc.gz.tar',
                    'filename': 'test_pc.gz.tar',
                    'md5': '52bbf03a7c50ee7152ccb9d357c2bb30',
                },
            ],
        )

    if request.param == 'CustomPrecomputedRMOnly':
        return DatasetDefinition(
            name='CustomPublicDataset',
            mirrors={
                'precomputed_reading_measures': (
                    'https://example.com/',
                    'https://another_example.com/',
                ),
            },
            resources=[
                {
                    'content': 'precomputed_reading_measures',
                    'url': 'test_rm.gz.tar',
                    'filename': 'test_rm.gz.tar',
                    'md5': '52bbf03a7c50ee7152ccb9d357c2bb30',
                },
            ],
        )

    if request.param == 'CustomPrecomputedRMOnlyNoMirror':
        return DatasetDefinition(
            name='CustomPublicDataset',
            resources=[
                {
                    'content': 'precomputed_reading_measures',
                    'url': 'https://example.com/test_rm.gz.tar',
                    'filename': 'test_rm.gz.tar',
                    'md5': '52bbf03a7c50ee7152ccb9d357c2bb30',
                },
            ],
        )

    assert False, f'unknown dataset_definition fixture {request.param}'


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
            DatasetPaths(root='/data/set/path'),
            {
                'root': Path('/data/set/path/'),
                'dataset': Path('/data/set/path/CustomPublicDataset'),
                'downloads': Path('/data/set/path/CustomPublicDataset/downloads'),
            },
            id='no_paths',
        ),
        pytest.param(
            DatasetPaths(root='/data/set/path', dataset='.'),
            {
                'root': Path('/data/set/path/'),
                'dataset': Path('/data/set/path/'),
                'downloads': Path('/data/set/path/downloads'),
            },
            id='dataset_dot',
        ),
        pytest.param(
            DatasetPaths(root='/data/set/path', dataset='dataset'),
            {
                'root': Path('/data/set/path/'),
                'dataset': Path('/data/set/path/dataset'),
                'downloads': Path('/data/set/path/dataset/downloads'),
            },
            id='explicit_dataset_dirname',
        ),
        pytest.param(
            DatasetPaths(root='/data/set/path', downloads='custom_downloads'),
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
    dataset = Dataset(dataset_definition, path=init_path)

    assert dataset.paths.root == expected_paths['root']
    assert dataset.paths.dataset == expected_paths['dataset']
    assert dataset.paths.downloads == expected_paths['downloads']


@mock.patch('pymovements.dataset.dataset_download.download_file')
@pytest.mark.filterwarnings('ignore:Failed to download from mirror.*:UserWarning')
@pytest.mark.parametrize('dataset_definition', ['CustomGazeOnly'], indirect=['dataset_definition'])
def test_dataset_download_both_mirrors_fail_gaze_only(
        mock_download_file,
        tmp_path,
        dataset_definition,
):
    paths = DatasetPaths(root=tmp_path, dataset='.')
    dataset = Dataset(dataset_definition, path=paths)

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
    'dataset_definition', ['CustomGazeOnlyNoMirror'], indirect=['dataset_definition'],
)
def test_dataset_download_without_mirrors_fail_gaze_only(
        mock_download_file,
        tmp_path,
        dataset_definition,
):
    paths = DatasetPaths(root=tmp_path, dataset='.')
    dataset = Dataset(dataset_definition, path=paths)

    mock_download_file.side_effect = OSError

    with pytest.raises(
        RuntimeError,
        match='downloading resource https://example.com/test.gz.tar failed.',
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
    ])


@mock.patch('pymovements.dataset.dataset_download.download_file')
@pytest.mark.filterwarnings('ignore:Failed to download from mirror.*:UserWarning')
@pytest.mark.parametrize(
    'dataset_definition', ['CustomPrecomputedOnly'], indirect=['dataset_definition'],
)
def test_dataset_download_precomputed_events_both_mirrors_fail(
        mock_download_file,
        tmp_path,
        dataset_definition,
):
    mock_download_file.side_effect = OSError()

    paths = DatasetPaths(root=tmp_path, dataset='.')
    dataset = Dataset(dataset_definition, path=paths)

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
    'dataset_definition', ['CustomPrecomputedOnlyNoMirror'], indirect=['dataset_definition'],
)
def test_dataset_download_precomputed_events_without_mirrors_fail(
        mock_download_file,
        tmp_path,
        dataset_definition,
):
    mock_download_file.side_effect = OSError()

    paths = DatasetPaths(root=tmp_path, dataset='.')
    dataset = Dataset(dataset_definition, path=paths)

    with pytest.raises(
        RuntimeError,
        match='downloading resource https://example.com/test_pc.gz.tar failed.',
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
    ])


@mock.patch('pymovements.dataset.dataset_download.download_file')
@pytest.mark.filterwarnings('ignore:Failed to download from mirror.*:UserWarning')
@pytest.mark.parametrize(
    'dataset_definition', ['CustomPrecomputedRMOnly'], indirect=['dataset_definition'],
)
def test_dataset_download_precomputed_reading_measures_both_mirrors_fail(
        mock_download_file,
        tmp_path,
        dataset_definition,
):
    mock_download_file.side_effect = OSError()

    paths = DatasetPaths(root=tmp_path, dataset='.')
    dataset = Dataset(dataset_definition, path=paths)

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
    'dataset_definition', ['CustomPrecomputedRMOnlyNoMirror'], indirect=['dataset_definition'],
)
def test_dataset_download_precomputed_reading_measures_without_mirrors_fail(
        mock_download_file,
        tmp_path,
        dataset_definition,
):
    mock_download_file.side_effect = OSError()

    paths = DatasetPaths(root=tmp_path, dataset='.')
    dataset = Dataset(dataset_definition, path=paths)

    with pytest.raises(
        RuntimeError,
        match='downloading resource https://example.com/test_rm.gz.tar failed.',
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
    ])


@mock.patch('pymovements.dataset.dataset_download.download_file')
@pytest.mark.filterwarnings('ignore:Failed to download from mirror.*:UserWarning')
@pytest.mark.parametrize(
    'dataset_definition', ['CustomGazeAndPrecomputed'], indirect=['dataset_definition'],
)
def test_dataset_download_precomputed_and_gaze_both_mirrors_fail(
        mock_download_file,
        tmp_path,
        dataset_definition,
):
    mock_download_file.side_effect = OSError()

    paths = DatasetPaths(root=tmp_path, dataset='.')
    dataset = Dataset(dataset_definition, path=paths)

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
    'dataset_definition', ['CustomGazeAndPrecomputedNoMirror'], indirect=['dataset_definition'],
)
def test_dataset_download_precomputed_and_gaze_without_mirrors_fail(
        mock_download_file,
        tmp_path,
        dataset_definition,
):
    mock_download_file.side_effect = OSError()

    paths = DatasetPaths(root=tmp_path, dataset='.')
    dataset = Dataset(dataset_definition, path=paths)

    with pytest.raises(
        RuntimeError,
        match='downloading resource https://example.com/test.gz.tar failed.',
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
    ])


@mock.patch('pymovements.dataset.dataset_download.download_file')
@pytest.mark.filterwarnings('ignore:Failed to download from mirror.*:UserWarning')
@pytest.mark.parametrize(
    'dataset_definition', ['CustomGazeOnly'], indirect=['dataset_definition'],
)
def test_dataset_download_first_mirror_gaze_fails(mock_download_file, tmp_path, dataset_definition):
    mock_download_file.side_effect = [OSError(), None]

    paths = DatasetPaths(root=tmp_path, dataset='.')
    dataset = Dataset(dataset_definition, path=paths)
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
@pytest.mark.filterwarnings('ignore:Failed to download from mirror.*:UserWarning')
@pytest.mark.parametrize(
    'dataset_definition', ['CustomPrecomputedOnly'], indirect=['dataset_definition'],
)
def test_dataset_download_first_mirror_precomputed_fails(
        mock_download_file, tmp_path, dataset_definition,
):
    mock_download_file.side_effect = [OSError(), None]

    paths = DatasetPaths(root=tmp_path, dataset='.')
    dataset = Dataset(dataset_definition, path=paths)
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
@pytest.mark.filterwarnings('ignore:Failed to download from mirror.*:UserWarning')
@pytest.mark.parametrize(
    'dataset_definition', ['CustomPrecomputedRMOnly'], indirect=['dataset_definition'],
)
def test_dataset_download_first_mirror_precomputed_fails_rm(
        mock_download_file, tmp_path, dataset_definition,
):
    mock_download_file.side_effect = [OSError(), None]

    paths = DatasetPaths(root=tmp_path, dataset='.')
    dataset = Dataset(dataset_definition, path=paths)
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
@pytest.mark.filterwarnings('ignore:Failed to download from mirror.*:UserWarning')
@pytest.mark.parametrize(
    'dataset_definition', ['CustomGazeAndPrecomputed'], indirect=['dataset_definition'],
)
def test_dataset_download_first_mirror_fails(mock_download_file, tmp_path, dataset_definition):
    mock_download_file.side_effect = [OSError(), None, OSError(), None]

    paths = DatasetPaths(root=tmp_path, dataset='.')
    dataset = Dataset(dataset_definition, path=paths)
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
@pytest.mark.filterwarnings('ignore:Failed to download from mirror.*:UserWarning')
@pytest.mark.parametrize(
    'dataset_definition',
    [
        'CustomGazeOnly', 'CustomGazeOnlyNoMirror',
        'CustomGazeAndPrecomputed', 'CustomGazeAndPrecomputedNoMirror',
    ],
    indirect=['dataset_definition'],
)
def test_dataset_download_file_not_found(mock_download_file, tmp_path, dataset_definition):
    mock_download_file.side_effect = RuntimeError()

    paths = DatasetPaths(root=tmp_path, dataset='.')
    dataset = Dataset(dataset_definition, path=paths)

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
@pytest.mark.filterwarnings('ignore:Failed to download from mirror.*:UserWarning')
@pytest.mark.parametrize(
    'dataset_definition',
    ['CustomPrecomputedOnly', 'CustomPrecomputedOnlyNoMirror'],
    indirect=['dataset_definition'],
)
def test_dataset_download_file_precomputed_not_found(
        mock_download_file, tmp_path, dataset_definition,
):
    mock_download_file.side_effect = RuntimeError()

    paths = DatasetPaths(root=tmp_path, dataset='.')
    dataset = Dataset(dataset_definition, path=paths)

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
    [
        'CustomGazeOnly', 'CustomGazeOnlyNoMirror',
        'CustomGazeAndPrecomputed', 'CustomGazeAndPrecomputedNoMirror',
    ],
    indirect=['dataset_definition'],
)
def test_dataset_download_no_extract(mock_download_file, tmp_path, dataset_definition):
    mock_download_file.return_value = 'path'

    paths = DatasetPaths(root=tmp_path, dataset='.')
    dataset = Dataset(dataset_definition, path=paths)
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
    ['CustomPrecomputedOnly', 'CustomPrecomputedOnlyNoMirror'],
    indirect=['dataset_definition'],
)
def test_dataset_download_precomputed_no_extract(mock_download_file, tmp_path, dataset_definition):
    mock_download_file.return_value = 'path'

    paths = DatasetPaths(root=tmp_path, dataset='.')
    dataset = Dataset(dataset_definition, path=paths)
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
    ['CustomPrecomputedRMOnly', 'CustomPrecomputedRMOnlyNoMirror'],
    indirect=['dataset_definition'],
)
def test_dataset_download_precomputed_no_extract_rm(
        mock_download_file, tmp_path, dataset_definition,
):
    mock_download_file.return_value = 'path'

    paths = DatasetPaths(root=tmp_path, dataset='.')
    dataset = Dataset(dataset_definition, path=paths)
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
@pytest.mark.parametrize(
    'dataset_definition',
    ['CustomGazeOnly', 'CustomGazeOnlyNoMirror'],
    indirect=['dataset_definition'],
)
def test_dataset_extract_remove_finished_true_gaze(
        mock_extract_archive,
        tmp_path,
        dataset_definition,
):
    mock_extract_archive.return_value = 'path'

    paths = DatasetPaths(root=tmp_path, dataset='.')
    dataset = Dataset(dataset_definition, path=paths)
    dataset.extract(remove_finished=True, remove_top_level=False, verbose=1)

    mock_extract_archive.assert_has_calls([
        mock.call(
            source_path=tmp_path / 'downloads' / 'test.gz.tar',
            destination_path=tmp_path / 'raw',
            recursive=True,
            remove_finished=True,
            remove_top_level=False,
            resume=True,
            verbose=1,
        ),
    ])


@mock.patch('pymovements.dataset.dataset_download.extract_archive')
@pytest.mark.parametrize(
    'dataset_definition',
    ['CustomPrecomputedRMOnly', 'CustomPrecomputedRMOnlyNoMirror'],
    indirect=['dataset_definition'],
)
def test_dataset_extract_rm(
        mock_extract_archive,
        tmp_path,
        dataset_definition,
):
    mock_extract_archive.return_value = 'path'

    paths = DatasetPaths(root=tmp_path, dataset='.')
    dataset = Dataset(dataset_definition, path=paths)
    dataset.extract(verbose=1)

    mock_extract_archive.assert_has_calls([
        mock.call(
            source_path=tmp_path / 'downloads' / 'test_rm.gz.tar',
            destination_path=tmp_path / 'precomputed_reading_measures',
            recursive=True,
            remove_finished=False,
            remove_top_level=True,
            resume=True,
            verbose=1,
        ),
    ])


@mock.patch('pymovements.dataset.dataset_download.extract_archive')
@pytest.mark.parametrize(
    'dataset_definition',
    ['CustomGazeAndPrecomputed', 'CustomGazeAndPrecomputedNoMirror'],
    indirect=['dataset_definition'],
)
def test_dataset_extract_remove_finished_true_both(
        mock_extract_archive,
        tmp_path,
        dataset_definition,
):
    mock_extract_archive.return_value = 'path'

    paths = DatasetPaths(root=tmp_path, dataset='.')
    dataset = Dataset(dataset_definition, path=paths)
    dataset.extract(remove_finished=True, remove_top_level=False, verbose=1)

    mock_extract_archive.assert_has_calls([
        mock.call(
            source_path=tmp_path / 'downloads' / 'test.gz.tar',
            destination_path=tmp_path / 'raw',
            recursive=True,
            remove_finished=True,
            remove_top_level=False,
            resume=True,
            verbose=1,
        ),
        mock.call(
            source_path=tmp_path / 'downloads' / 'test_pc.gz.tar',
            destination_path=tmp_path / 'precomputed_events',
            recursive=True,
            remove_finished=True,
            remove_top_level=False,
            resume=True,
            verbose=1,
        ),
    ])


@mock.patch('pymovements.dataset.dataset_download.extract_archive')
@pytest.mark.parametrize(
    'dataset_definition',
    ['CustomPrecomputedOnly', 'CustomPrecomputedOnlyNoMirror'],
    indirect=['dataset_definition'],
)
def test_dataset_extract_remove_finished_true_precomputed(
        mock_extract_archive,
        tmp_path,
        dataset_definition,
):
    mock_extract_archive.return_value = 'path'

    paths = DatasetPaths(root=tmp_path, dataset='.')
    dataset = Dataset(dataset_definition, path=paths)
    dataset.extract(remove_finished=True, remove_top_level=False, verbose=1)

    mock_extract_archive.assert_has_calls([
        mock.call(
            source_path=tmp_path / 'downloads' / 'test_pc.gz.tar',
            destination_path=tmp_path / 'precomputed_events',
            recursive=True,
            remove_finished=True,
            remove_top_level=False,
            resume=True,
            verbose=1,
        ),
    ])


@mock.patch('pymovements.dataset.dataset_download.extract_archive')
@pytest.mark.parametrize(
    'dataset_definition',
    ['CustomGazeAndPrecomputed', 'CustomGazeAndPrecomputedNoMirror'],
    indirect=['dataset_definition'],
)
def test_dataset_extract_remove_finished_false_both(
        mock_extract_archive,
        tmp_path,
        dataset_definition,
):
    mock_extract_archive.return_value = 'path'

    paths = DatasetPaths(root=tmp_path, dataset='.')
    dataset = Dataset(dataset_definition, path=paths)
    dataset.extract()

    mock_extract_archive.assert_has_calls([
        mock.call(
            source_path=tmp_path / 'downloads' / 'test.gz.tar',
            destination_path=tmp_path / 'raw',
            recursive=True,
            remove_finished=False,
            remove_top_level=True,
            resume=True,
            verbose=1,
        ),
        mock.call(
            source_path=tmp_path / 'downloads' / 'test_pc.gz.tar',
            destination_path=tmp_path / 'precomputed_events',
            recursive=True,
            remove_finished=False,
            remove_top_level=True,
            resume=True,
            verbose=1,
        ),
    ])


@mock.patch('pymovements.dataset.dataset_download.extract_archive')
@pytest.mark.parametrize(
    'dataset_definition',
    ['CustomGazeOnly', 'CustomGazeOnlyNoMirror'],
    indirect=['dataset_definition'],
)
def test_dataset_extract_remove_finished_false_gaze(
        mock_extract_archive,
        tmp_path,
        dataset_definition,
):
    mock_extract_archive.return_value = 'path'

    paths = DatasetPaths(root=tmp_path, dataset='.')
    dataset = Dataset(dataset_definition, path=paths)
    dataset.extract()

    mock_extract_archive.assert_has_calls([
        mock.call(
            source_path=tmp_path / 'downloads' / 'test.gz.tar',
            destination_path=tmp_path / 'raw',
            recursive=True,
            remove_finished=False,
            remove_top_level=True,
            resume=True,
            verbose=1,
        ),
    ])


@mock.patch('pymovements.dataset.dataset_download.extract_archive')
@pytest.mark.parametrize(
    'dataset_definition',
    ['CustomPrecomputedOnly', 'CustomPrecomputedOnlyNoMirror'],
    indirect=['dataset_definition'],
)
def test_dataset_extract_remove_finished_false_precomputed(
        mock_extract_archive,
        tmp_path,
        dataset_definition,
):
    mock_extract_archive.return_value = 'path'

    paths = DatasetPaths(root=tmp_path, dataset='.')
    dataset = Dataset(dataset_definition, path=paths)
    dataset.extract()

    mock_extract_archive.assert_has_calls([
        mock.call(
            source_path=tmp_path / 'downloads' / 'test_pc.gz.tar',
            destination_path=tmp_path / 'precomputed_events',
            recursive=True,
            remove_finished=False,
            remove_top_level=True,
            resume=True,
            verbose=1,
        ),
    ])


@mock.patch('pymovements.dataset.dataset_download.download_file')
@mock.patch('pymovements.dataset.dataset_download.extract_archive')
@pytest.mark.parametrize(
    'dataset_definition',
    ['CustomGazeAndPrecomputed', 'CustomGazeAndPrecomputedNoMirror'],
    indirect=['dataset_definition'],
)
def test_dataset_download_default_extract_both(
        mock_extract, mock_download, tmp_path, dataset_definition,
):
    mock_extract.return_value = None
    mock_download.return_value = None

    Dataset(dataset_definition, path=tmp_path).download()


@mock.patch('pymovements.dataset.dataset_download.download_file')
@mock.patch('pymovements.dataset.dataset_download.extract_archive')
@pytest.mark.parametrize(
    'dataset_definition',
    ['CustomGazeOnly', 'CustomGazeOnlyNoMirror'],
    indirect=['dataset_definition'],
)
def test_dataset_download_default_extract_gaze(
        mock_extract, mock_download, tmp_path, dataset_definition,
):
    mock_extract.return_value = None
    mock_download.return_value = None

    Dataset(dataset_definition, path=tmp_path).download()

    mock_download.assert_called_once()
    mock_extract.assert_called_once()


@mock.patch('pymovements.dataset.dataset_download.download_file')
@mock.patch('pymovements.dataset.dataset_download.extract_archive')
@pytest.mark.parametrize(
    'dataset_definition',
    ['CustomPrecomputedOnly', 'CustomPrecomputedOnlyNoMirror'],
    indirect=['dataset_definition'],
)
def test_dataset_download_default_extract_precomputed(
        mock_extract, mock_download, tmp_path, dataset_definition,
):
    mock_extract.return_value = None
    mock_download.return_value = None

    Dataset(dataset_definition, path=tmp_path).download()

    mock_download.assert_called_once()
    mock_extract.assert_called_once()


@pytest.mark.parametrize(
    ('dataset_definition', 'expected_exception', 'expected_msg_prefix'),
    [
        pytest.param(
            DatasetDefinition(
                name='CustomPublicDataset',
                mirrors={'gaze': ['https://example.com/']},
                resources=[
                    {
                        'content': 'gaze',
                        'url': None,
                        'filename': 'test.gz.tar',
                        'md5': '52bbf03a7c50ee7152ccb9d357c2bb30',
                    },
                ],
            ),
            AttributeError,
            'Resource.url must not be None',
            id='mirrors_url_none',
        ),
        pytest.param(
            DatasetDefinition(
                name='CustomPublicDataset',
                mirrors={'gaze': ['https://example.com/']},
                resources=[
                    {
                        'content': 'gaze',
                        'url': 'https://example.com/test.gz.tar',
                        'filename': None,
                        'md5': '52bbf03a7c50ee7152ccb9d357c2bb30',
                    },
                ],
            ),
            AttributeError,
            'Resource.filename must not be None',
            id='mirrors_filename_none',
        ),
        pytest.param(
            DatasetDefinition(
                name='CustomPublicDataset',
                resources=[{
                    'content': 'gaze',
                    'url': None,
                    'filename': 'test.gz.tar',
                    'md5': '52bbf03a7c50ee7152ccb9d357c2bb30',
                }],
            ),
            AttributeError,
            'Resource.url must not be None',
            id='no_mirrors_url_none',
        ),
        pytest.param(
            DatasetDefinition(
                name='CustomPublicDataset',
                resources=[{
                    'content': 'gaze',
                    'url': 'https://example.com/test.gz.tar',
                    'filename': None,
                    'md5': '52bbf03a7c50ee7152ccb9d357c2bb30',
                }],
            ),
            AttributeError,
            'Resource.filename must not be None',
            id='no_mirrors_filename_none',
        ),
        pytest.param(
            DatasetDefinition(
                name='CustomPublicDataset',
                resources=[{
                    'content': 'gaze',
                    'url': 'test.gz.tar',
                    'filename': 'test.gz.tar',
                    'md5': '52bbf03a7c50ee7152ccb9d357c2bb30',
                }],
            ),
            ValueError,
            'unknown url type: ',
            id='no_mirrors_no_http_resource_gaze',
        ),
        pytest.param(
            DatasetDefinition(
                name='CustomPublicDataset',
                resources=[{
                    'content': 'gaze',
                    'url': 'test.gz.tar',
                    'filename': 'test.gz.tar',
                    'md5': '52bbf03a7c50ee7152ccb9d357c2bb30',
                }],
            ),
            ValueError,
            'unknown url type: ',
            id='no_mirrors_no_http_resource_events',
        ),
        pytest.param(
            DatasetDefinition(
                name='CustomPublicDataset',
                resources=[{
                    'content': 'gaze',
                    'url': 'test.gz.tar',
                    'filename': 'test.gz.tar',
                    'md5': '52bbf03a7c50ee7152ccb9d357c2bb30',
                }],
            ),
            ValueError,
            'unknown url type: ',
            id='no_mirrors_no_http_resource_measures',
        ),
    ],
)
def test_dataset_download_raises_exception(
        dataset_definition, expected_exception, expected_msg_prefix, tmp_path,
):
    with pytest.raises(expected_exception) as excinfo:
        Dataset(dataset_definition, path=tmp_path).download()
    msg, = excinfo.value.args
    assert msg.startswith(expected_msg_prefix)


@pytest.mark.parametrize(
    'definition',
    [
        pytest.param(
            DatasetDefinition(
                name='CustomPublicDataset',
                mirrors={
                    'gaze': (
                        'https://example.com/',
                        'https://another_example.com/',
                    ),
                },
            ),
            id='gaze',
        ),
        pytest.param(
            DatasetDefinition(
                name='CustomPublicDataset',
                mirrors={
                    'precomputed_events': (
                        'https://example.com/',
                        'https://another_example.com/',
                    ),
                },
            ),
            id='precomputed_events',
        ),
        pytest.param(
            DatasetDefinition(
                name='CustomPublicDataset',
                mirrors={
                    'precomputed_reading_measures': (
                        'https://example.com/',
                        'https://another_example.com/',
                    ),
                },
            ),
            id='precomputed_reading_measures',
        ),
    ],
)
def test_dataset_download_no_resources_raises_exception(definition, tmp_path):
    with pytest.raises(AttributeError) as excinfo:
        Dataset(definition, path=tmp_path).download()

    msg, = excinfo.value.args

    expected_msg = 'resources must be specified to download a dataset.'
    assert msg == expected_msg


def test_public_dataset_registered_correct_attributes(tmp_path, dataset_definition):
    dataset = Dataset(dataset_definition, path=tmp_path)

    assert dataset.definition.mirrors == dataset_definition.mirrors
    assert dataset.definition.resources == dataset_definition.resources
    assert dataset.definition.experiment == dataset_definition.experiment


def test_extract_dataset_precomputed_move_single_file(tmp_path, testfiles_dirpath):
    definition = DatasetDefinition(
        name='CustomPublicDataset',
        resources=[
            {
                'content': 'precomputed_events',
                'filename': '18sat_fixfinal.csv',
            },
        ],
    )

    # Create directory and copy test file.
    (tmp_path / 'downloads').mkdir(parents=True)
    shutil.copyfile(
        testfiles_dirpath / '18sat_fixfinal.csv',
        tmp_path / 'downloads' / '18sat_fixfinal.csv',
    )

    Dataset(definition, path=tmp_path).extract()


def test_extract_dataset_precomputed_rm_move_single_file(tmp_path, testfiles_dirpath):
    definition = DatasetDefinition(
        name='CustomPublicDataset',
        resources=[
            {
                'content': 'precomputed_reading_measures',
                'filename': 'copco_rm_dummy.csv',
            },
        ],
    )

    # Create directory and copy test file.
    (tmp_path / 'downloads').mkdir(parents=True)

    shutil.copyfile(
        testfiles_dirpath / 'copco_rm_dummy.csv',
        tmp_path / 'downloads' / 'copco_rm_dummy.csv',
    )

    Dataset(definition, path=tmp_path).extract()
