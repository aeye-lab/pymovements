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

    pymovements.datasets.Dataset
    pymovements.datasets.PublicDataset
    pymovements.datasets.PublicDatasetDefinition

.. rubric:: Public Dataset Definitions

.. autosummary::
   :toctree:
   :template: class.rst

    pymovements.datasets.definitions.ToyDataset
    pymovements.datasets.definitions.GazeBase
    pymovements.datasets.definitions.JuDo1000
"""
from pymovements.datasets import definitions  # noqa:F401
from pymovements.datasets.dataset import Dataset  # noqa: F401
from pymovements.datasets.definitions import GazeBase  # noqa: F401
from pymovements.datasets.definitions import JuDo1000  # noqa: F401
from pymovements.datasets.definitions import ToyDataset  # noqa: F401
from pymovements.datasets.public_dataset import PublicDataset  # noqa: F401
from pymovements.datasets.public_dataset import PublicDatasetDefinition  # noqa: F401
