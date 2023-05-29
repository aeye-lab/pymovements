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
   pymovements.gaze.transforms.pos2acc
   pymovements.gaze.transforms.pos2vel
   pymovements.gaze.transforms.norm
   pymovements.gaze.transforms.split
   pymovements.gaze.transforms.downsample
   pymovements.gaze.transforms.consecutive

.. rubric:: Transformations

.. autosummary::
   :toctree:

   pymovements.gaze.transforms_pl.center_origin
   pymovements.gaze.transforms_pl.downsample
   pymovements.gaze.transforms_pl.norm
   pymovements.gaze.transforms_pl.pix2deg
   pymovements.gaze.transforms_pl.pos2acc
   pymovements.gaze.transforms_pl.pos2vel
   pymovements.gaze.transforms_pl.savitzky_golay
"""
from pymovements.gaze import transforms
from pymovements.gaze import transforms_pl
from pymovements.gaze.experiment import Experiment
from pymovements.gaze.from_numpy import from_numpy
from pymovements.gaze.from_pandas import from_pandas
from pymovements.gaze.gaze_dataframe import GazeDataFrame
from pymovements.gaze.screen import Screen


__all__ = [
    'Experiment',
    'from_numpy',
    'from_pandas',
    'GazeDataFrame',
    'Screen',
    'transforms',
    'transforms_pl',
]
