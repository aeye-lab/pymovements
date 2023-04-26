# Copyright (c) 2022-2023 The pymovements Project Authors
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
"""This module provides access to dataset classes.

.. rubric:: Classes

.. autosummary::
   :toctree:
   :template: class.rst

    pymovements.dataset.Dataset
    pymovements.dataset.DatasetDefinition
    pymovements.dataset.DatasetLibrary
    pymovements.dataset.DatasetPaths
"""
from pymovements.dataset.dataset import Dataset
from pymovements.dataset.dataset_definition import DatasetDefinition
from pymovements.dataset.dataset_library import DatasetLibrary
from pymovements.dataset.dataset_library import register_dataset
from pymovements.dataset.dataset_paths import DatasetPaths


__all__ = [
    'Dataset',
    'DatasetDefinition',
    'DatasetLibrary',
    'DatasetPaths',
    'register_dataset',
]
