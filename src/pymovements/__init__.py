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
"""This module provides top-level access to all submodules."""
from pymovements import datasets
from pymovements import events
from pymovements import gaze
from pymovements import plotting
from pymovements import synthetic
from pymovements import utils
from pymovements.dataset import Dataset
from pymovements.dataset import DatasetDefinition
from pymovements.dataset import DatasetLibrary
from pymovements.dataset import DatasetPaths
from pymovements.dataset import register_dataset
from pymovements.gaze.experiment import Experiment
from pymovements.gaze.screen import Screen


__all__ = [
    'Dataset',
    'DatasetDefinition',
    'DatasetLibrary',
    'DatasetPaths',
    'datasets',
    'events',
    'Experiment',
    'gaze',
    'plotting',
    'register_dataset',
    'Screen',
    'synthetic',
    'utils',
]
