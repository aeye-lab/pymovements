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
"""This module holds gaze time series related functionality.

.. rubric:: Classes

.. autosummary::
   :toctree:
   :template: class.rst

    pymovements.gaze.Experiment
    pymovements.gaze.Screen
    pymovements.gaze.GazeDataFrame

.. rubric:: Transformations

.. autosummary::
   :toctree:

   pymovements.gaze.transforms.pix2deg
   pymovements.gaze.transforms.pos2vel
   pymovements.gaze.transforms.norm
   pymovements.gaze.transforms.split
   pymovements.gaze.transforms.downsample
   pymovements.gaze.transforms.consecutive
"""
from pymovements.gaze import transforms  # noqa: F401
from pymovements.gaze.experiment import Experiment
from pymovements.gaze.gaze_dataframe import GazeDataFrame  # noqa: F401
from pymovements.gaze.screen import Screen


__all__ = [
    'Experiment',
    'GazeDataFrame',
    'Screen',
    'transforms',
]
