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
"""Test public dataset definitions."""
from __future__ import annotations

from pathlib import Path

import pytest

import pymovements as pm


@pytest.mark.parametrize(
    ('public_dataset', 'dataset_name'),
    # please add datasets in alphabetical order
    [
        pytest.param(pm.datasets.BSC, 'BSC', id='BSC'),
        pytest.param(pm.datasets.BSCII, 'BSCII', id='BSCII'),
        pytest.param(pm.datasets.CodeComprehension, 'CodeComprehension', id='CodeComprehension'),
        pytest.param(pm.datasets.CopCo, 'CopCo', id='CopCo'),
        pytest.param(pm.datasets.DAEMONS, 'DAEMONS', id='DAEMONS'),
        pytest.param(pm.datasets.DIDEC, 'DIDEC', id='DIDEC'),
        pytest.param(pm.datasets.EMTeC, 'EMTeC', id='EMTeC'),
        pytest.param(pm.datasets.FakeNewsPerception, 'FakeNewsPerception', id='FakeNewsPerception'),
        pytest.param(pm.datasets.GazeBase, 'GazeBase', id='GazeBase'),
        pytest.param(pm.datasets.GazeBaseVR, 'GazeBaseVR', id='GazeBaseVR'),
        pytest.param(pm.datasets.GazeOnFaces, 'GazeOnFaces', id='GazeOnFaces'),
        pytest.param(pm.datasets.HBN, 'HBN', id='HBN'),
        pytest.param(pm.datasets.InteRead, 'InteRead', id='InteRead'),
        pytest.param(pm.datasets.JuDo1000, 'JuDo1000', id='JuDo1000'),
        pytest.param(pm.datasets.PoTeC, 'PoTeC', id='PoTeC'),
        pytest.param(
            pm.datasets.PotsdamBingeRemotePVT,
            'PotsdamBingeRemotePVT',
            id='PotsdamBingeRemotePVT',
        ),
        pytest.param(
            pm.datasets.PotsdamBingeWearablePVT,
            'PotsdamBingeWearablePVT',
            id='PotsdamBingeWearablePVT',
        ),
        pytest.param(pm.datasets.Provo, 'Provo', id='Provo'),
        pytest.param(pm.datasets.SBSAT, 'SBSAT', id='SBSAT'),
        pytest.param(pm.datasets.ToyDataset, 'ToyDataset', id='ToyDataset'),
        pytest.param(pm.datasets.ToyDatasetEyeLink, 'ToyDatasetEyeLink', id='ToyDatasetEyeLink'),
        pytest.param(pm.datasets.UCL, 'UCL', id='UCL'),
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
def test_public_dataset_registered(public_dataset, dataset_name, dataset_path, downloads, str_root):
    assert dataset_name in pm.DatasetLibrary.definitions
    assert pm.DatasetLibrary.get(dataset_name) == public_dataset
    assert pm.DatasetLibrary.get(dataset_name)().name == dataset_name

    dataset_definition = public_dataset()
    registered_definition = pm.DatasetLibrary.get(dataset_definition.name)()
    assert dataset_definition.has_files['gaze'] == registered_definition.has_files['gaze']
    assert dataset_definition.has_files['precomputed_events'] == registered_definition.has_files['precomputed_events']  # noqa: E501
    if dataset_definition.has_files['gaze']:
        assert dataset_definition.mirrors['gaze'] == registered_definition.mirrors['gaze']
        assert dataset_definition.resources['gaze'] == registered_definition.resources['gaze']
        assert dataset_definition.experiment == registered_definition.experiment
        assert dataset_definition.filename_format['gaze'] == registered_definition.filename_format['gaze']  # noqa: E501
        assert dataset_definition.filename_format_schema_overrides['gaze'] == registered_definition.filename_format_schema_overrides['gaze']  # noqa: E501
        assert dataset_definition.custom_read_kwargs['gaze'] == registered_definition.custom_read_kwargs['gaze']  # noqa: E501

    if dataset_definition.has_files['precomputed_events']:
        assert dataset_definition.mirrors['precomputed_events'] == registered_definition.mirrors['precomputed_events']  # noqa: E501
        assert dataset_definition.resources['precomputed_events'] == registered_definition.resources['precomputed_events']  # noqa: E501
        assert dataset_definition.experiment == registered_definition.experiment
        assert dataset_definition.filename_format['precomputed_events'] == registered_definition.filename_format['precomputed_events']  # noqa: E501
        assert dataset_definition.filename_format_schema_overrides['precomputed_events'] == registered_definition.filename_format_schema_overrides['precomputed_events']  # noqa: E501
        assert dataset_definition.custom_read_kwargs['precomputed_events'] == registered_definition.custom_read_kwargs['precomputed_events']  # noqa: E501

    if dataset_definition.has_files['precomputed_reading_measures']:
        assert dataset_definition.mirrors['precomputed_reading_measures'] == registered_definition.mirrors['precomputed_reading_measures']  # noqa: E501
        assert dataset_definition.resources['precomputed_reading_measures'] == registered_definition.resources['precomputed_reading_measures']  # noqa: E501
        assert dataset_definition.experiment == registered_definition.experiment
        assert dataset_definition.filename_format['precomputed_reading_measures'] == registered_definition.filename_format['precomputed_reading_measures']  # noqa: E501
        assert dataset_definition.filename_format_schema_overrides['precomputed_reading_measures'] == registered_definition.filename_format_schema_overrides['precomputed_reading_measures']  # noqa: E501
        assert dataset_definition.custom_read_kwargs['precomputed_reading_measures'] == registered_definition.custom_read_kwargs['precomputed_reading_measures']  # noqa: E501

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
