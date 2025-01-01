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
"""Provides dataset definitions.

.. rubric:: Dataset Definitions

.. autosummary::
   :toctree:
   :template: class.rst

    pymovements.datasets.BSC
    pymovements.datasets.CodeComprehension
    pymovements.datasets.CopCo
    pymovements.datasets.DIDEC
    pymovements.datasets.EMTeC
    pymovements.datasets.GazeBase
    pymovements.datasets.GazeBaseVR
    pymovements.datasets.GazeGraph
    pymovements.datasets.GazeOnFaces
    pymovements.datasets.HBN
    pymovements.datasets.InteRead
    pymovements.datasets.JuDo1000
    pymovements.datasets.PoTeC
    pymovements.datasets.SBSAT


.. rubric:: Example Datasets

.. autosummary::
   :toctree:
   :template: class.rst

    pymovements.datasets.ToyDataset
    pymovements.datasets.ToyDatasetEyeLink
"""
from pymovements.datasets.bsc import BSC
from pymovements.datasets.codecomprehension import CodeComprehension
from pymovements.datasets.copco import CopCo
from pymovements.datasets.didec import DIDEC
from pymovements.datasets.emtec import EMTeC
from pymovements.datasets.fakenews import FakeNewsPerception
from pymovements.datasets.gaze_graph import GazeGraph
from pymovements.datasets.gaze_on_faces import GazeOnFaces
from pymovements.datasets.gazebase import GazeBase
from pymovements.datasets.gazebasevr import GazeBaseVR
from pymovements.datasets.hbn import HBN
from pymovements.datasets.interead import InteRead
from pymovements.datasets.judo1000 import JuDo1000
from pymovements.datasets.potec import PoTeC
from pymovements.datasets.sb_sat import SBSAT
from pymovements.datasets.toy_dataset import ToyDataset
from pymovements.datasets.toy_dataset_eyelink import ToyDatasetEyeLink


__all__ = [
    'BSC',
    'CodeComprehension',
    'CopCo',
    'DIDEC',
    'EMTeC',
    'FakeNewsPerception',
    'GazeBase',
    'GazeBaseVR',
    'GazeGraph',
    'GazeOnFaces',
    'HBN',
    'InteRead',
    'JuDo1000',
    'PoTeC',
    'SBSAT',
    'ToyDataset',
    'ToyDatasetEyeLink',
]
