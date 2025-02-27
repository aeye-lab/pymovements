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

import pytest

import pymovements as pm


@pytest.mark.parametrize(
    ('dataset_name'),
    # please add datasets in alphabetical order
    [
        pytest.param('BSC', id='BSC'),
        pytest.param('BSCII', id='BSCII'),
        pytest.param('CodeComprehension', id='CodeComprehension'),
        pytest.param('CopCo', id='CopCo'),
        pytest.param('DIDEC', id='DIDEC'),
        pytest.param('EMTeC', id='EMTeC'),
        pytest.param('FakeNewsPerception', id='FakeNewsPerception'),
        pytest.param('GazeBase', id='GazeBase'),
        pytest.param('GazeBaseVR', id='GazeBaseVR'),
        pytest.param('GazeOnFaces', id='GazeOnFaces'),
        pytest.param('HBN', id='HBN'),
        pytest.param('InteRead', id='InteRead'),
        pytest.param('JuDo1000', id='JuDo1000'),
        pytest.param('PoTeC', id='PoTeC'),
        pytest.param('SBSAT', id='SBSAT'),
        pytest.param('ToyDataset', id='ToyDataset'),
        pytest.param('ToyDatasetEyeLink', id='ToyDatasetEyeLink'),
    ],
)
def test_public_dataset_registered(dataset_name):
    assert dataset_name in pm.DatasetLibrary.definitions
    assert pm.DatasetLibrary.get(dataset_name).name == dataset_name
