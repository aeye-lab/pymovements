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


def test_library_not_empty():
    assert len(pm.DatasetLibrary.definitions) >= 0


def test_list_names_is_list_of_str():
    names = pm.DatasetLibrary.names

    assert isinstance(names, list)

    for name in names:
        assert isinstance(name, str)
