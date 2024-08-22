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
"""Test all functionality in public datasets."""
from pathlib import Path

import pytest

import pymovements as pm


def construct_public_dataset(
        public_dataset,
        dataset_path,
        downloads,
        str_root,
):
    expected = {}
    expected['root'] = Path('/data/set/path')

    if str_root:
        init_path = '/data/set/path'
        expected['dataset'] = Path('/data/set/path')
        expected['downloads'] = Path('/data/set/path/downloads')

        dataset = pm.Dataset(public_dataset, path=init_path)
        return dataset, expected
    init_path = pm.DatasetPaths(
        root='/data/set/path',
        dataset=dataset_path,
        downloads=downloads,
    )

    if dataset_path == '.':
        expected['dataset'] = Path('/data/set/path')
    elif dataset_path == 'dataset_path':
        expected['dataset'] = Path('/data/set/path/dataset_path')
    else:
        expected['dataset'] = Path(f'/data/set/path/{public_dataset.__name__}')
    expected['downloads'] = expected['dataset'] / Path(downloads)

    dataset = pm.Dataset(public_dataset, path=init_path)
    return dataset, expected


@pytest.mark.parametrize(
    ('public_dataset'),
    [
        pytest.param(
            pm.datasets.SBSAT,
            id='Datasets_SBSAT',
        ),
        pytest.param(
            pm.datasets.GazeBase,
            id='Datasets_GazeBase',
        ),
        pytest.param(
            pm.datasets.GazeBaseVR,
            id='Datasets_GazeBaseVR',
        ),
        pytest.param(
            pm.datasets.JuDo1000,
            id='Datasets_JuDo1000',
        ),
        pytest.param(
            pm.datasets.PoTeC,
            id='Datasets_PoTeC',
        ),
        pytest.param(
            pm.datasets.HBN,
            id='Datasets_HBN',
        ),
        pytest.param(
            pm.datasets.GazeOnFaces,
            id='Datasets_GazeOnFaces',
        ),
        pytest.param(
            pm.datasets.ToyDataset,
            id='Datasets_ToyDataset',
        ),
        pytest.param(
            pm.datasets.ToyDatasetEyeLink,
            id='Datasets_ToyDatasetEyeLink',
        ),
    ],
)
@pytest.mark.parametrize(
    ('dataset_path'),
    [
        pytest.param(
            None,
            id='dataset_path_None',
        ),
        pytest.param(
            '.',
            id='dataset_path_dot',
        ),
        pytest.param(
            'dataset_path',
            id='dataset_path_dataset',
        ),
    ],
)
@pytest.mark.parametrize(
    ('downloads'),
    [
        pytest.param(
            'downloads',
            id='downloads_None',
        ),
        pytest.param(
            'custom_downloads',
            id='downloads_custom_downloads',
        ),

    ],
)
@pytest.mark.parametrize(
    ('str_root'),
    [
        pytest.param(
            True,
            id='path_str',
        ),
        pytest.param(
            False,
            id='path_DatasetPaths',
        ),
    ],
)
def test_paths(public_dataset, dataset_path, downloads, str_root):
    dataset, expected_paths = construct_public_dataset(
        public_dataset,
        dataset_path,
        downloads,
        str_root,
    )

    assert dataset.paths.root == expected_paths['root']
    assert dataset.path == expected_paths['dataset']
    assert dataset.paths.dataset == expected_paths['dataset']
    assert dataset.paths.downloads == expected_paths['downloads']
