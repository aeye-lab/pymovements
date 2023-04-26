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
"""Module to create a GazeDataFrame from a numpy array."""
from __future__ import annotations

import numpy as np
import polars as pl
from typing_extensions import Literal

from pymovements.gaze.experiment import Experiment
from pymovements.gaze.gaze_dataframe import GazeDataFrame


def from_numpy(
        data: np.ndarray,
        schema: list[str],
        experiment: Experiment | None = None,
        orient: Literal['col', 'row'] = 'col',
) -> GazeDataFrame:
    """Construct a :py:class:`~pymovements.gaze.gaze_dataframe.GazeDataFrame`
    from a numpy array.

    Parameters
    ----------
    data:
        Two-dimensional data represented as a numpy ndarray.
    schema:
        A list of column names.
    orient:
        Whether to interpret the two-dimensional data as columns or as rows.
    experiment : Experiment
        The experiment definition.

    Returns
    -------
    py:class:`~pymovements.GazeDataFrame`
    """
    df = pl.from_numpy(data=data, schema=schema, orient=orient)
    return GazeDataFrame(data=df, experiment=experiment)
