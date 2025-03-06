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
"""Test dataset library."""
from unittest import mock

import pytest

import pymovements as pm


def test_add_single_defintion():
    class CustomDatasetDefinition(pm.DatasetDefinition):
        name: str = 'CustomDatasetDefinition'

    pm.DatasetLibrary.add(CustomDatasetDefinition)
    definition = pm.DatasetLibrary.get('CustomDatasetDefinition')

    assert definition == CustomDatasetDefinition


def test_add_two_defintions():
    class CustomDatasetDefinition1(pm.DatasetDefinition):
        name: str = 'CustomDatasetDefinition1'

    class CustomDatasetDefinition2(pm.DatasetDefinition):
        name: str = 'CustomDatasetDefinition2'

    pm.DatasetLibrary.add(CustomDatasetDefinition1)
    pm.DatasetLibrary.add(CustomDatasetDefinition2)

    definition1 = pm.DatasetLibrary.get('CustomDatasetDefinition1')
    definition2 = pm.DatasetLibrary.get('CustomDatasetDefinition2')

    assert definition1 == CustomDatasetDefinition1
    assert definition2 == CustomDatasetDefinition2


def test_raise_value_error_get_non_existent_dataset():
    with pytest.raises(KeyError) as exc_info:
        pm.DatasetLibrary.get('NonExistent')

    msg, = exc_info.value.args
    error_msg_snippets = [
        'NonExistent',
        'not found in library',
        'Available datasets',
    ]
    for snippet in error_msg_snippets:
        assert snippet in msg


def test__add_shipped_datasets():
    with mock.patch('pymovements.dataset.dataset_library._add_shipped_datasets') as mock_add:
        pm.dataset.dataset_library._add_shipped_datasets()
        mock_add.assert_called_once()
