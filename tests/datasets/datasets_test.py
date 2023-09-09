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
"""Test public dataset definitions."""
from __future__ import annotations

import pytest

import pymovements as pm


@pytest.mark.parametrize(
    ('definition_class', 'dataset_name'),
    [
        pytest.param(pm.datasets.ToyDataset, 'ToyDataset', id='ToyDataset'),
        pytest.param(pm.datasets.GazeBase, 'GazeBase', id='GazeBase'),
        pytest.param(pm.datasets.GazeBaseVR, 'GazeBaseVR', id='GazeBaseVR'),
        pytest.param(pm.datasets.JuDo1000, 'JuDo1000', id='JuDo1000'),
    ],
)
def test_public_dataset_registered(definition_class, dataset_name):
    assert dataset_name in pm.DatasetLibrary.definitions
    assert pm.DatasetLibrary.get(dataset_name) == definition_class
    assert pm.DatasetLibrary.get(dataset_name)().name == dataset_name


@pytest.mark.parametrize(
    'dataset_definition_class',
    [
        pytest.param(pm.datasets.ToyDataset, id='ToyDataset'),
        pytest.param(pm.datasets.GazeBase, id='GazeBase'),
        pytest.param(pm.datasets.GazeBaseVR, id='GazeBaseVR'),
        pytest.param(pm.datasets.JuDo1000, id='JuDo1000'),
    ],
)
def test_public_dataset_registered_correct_attributes(dataset_definition_class):
    dataset_definition = dataset_definition_class()

    registered_definition = pm.DatasetLibrary.get(dataset_definition.name)()

    assert dataset_definition.mirrors == registered_definition.mirrors
    assert dataset_definition.resources == registered_definition.resources
    assert dataset_definition.experiment == registered_definition.experiment
    assert dataset_definition.filename_format == registered_definition.filename_format
    assert dataset_definition.filename_format_dtypes == registered_definition.filename_format_dtypes
    assert dataset_definition.custom_read_kwargs == registered_definition.custom_read_kwargs
