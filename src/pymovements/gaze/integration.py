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

from typing import Literal

import numpy as np
import pandas as pd
import polars as pl

from pymovements.gaze.experiment import Experiment
from pymovements.gaze.gaze_dataframe import GazeDataFrame
from pymovements.utils import checks


def from_numpy(
        data: np.ndarray = None,
        time: np.ndarray | None = None,
        pixel: np.ndarray | list[np.ndarray] | None = None,
        position: np.ndarray | list[np.ndarray] | None = None,
        velocity: np.ndarray | list[np.ndarray] | None = None,
        acceleration: np.ndarray | list[np.ndarray] | None = None,
        schema: list[str] = None,
        experiment: Experiment | None = None,
        orient: Literal['col', 'row'] = 'col',
        time_column: str | None = None,
        pixel_columns: list[str] | list[int] | None = None,
        position_columns: list[str] | list[int] | None = None,
        velocity_columns: list[str] | list[int] | None = None,
        acceleration_columns: list[str] | list[int] | None = None,
) -> GazeDataFrame:
    """Construct a :py:class:`~pymovements.gaze.gaze_dataframe.GazeDataFrame`
    from a numpy array.

    Parameters
    ----------
    data:
        Two-dimensional data represented as a numpy ndarray.
    time:
        Array of timestamps.
    pixel:
        Array of gaze pixel positions.
    position:
        Array of gaze positions in degrees of visual angle.
    velocity:
        Array of gaze velocities in degrees of visual angle per second.
    acceleration:
        Array of gaze accelerations in degrees of visual angle per square second.
    schema:
        A list of column names.
    orient:
        Whether to interpret the two-dimensional data as columns or as rows.
    experiment : Experiment
        The experiment definition.
            time_column: str | None = None,
    time_column:
        The name of the timestamp column in the input data frame.
    pixel_columns:
        The name of the pixel position columns in the input data frame.
    position_columns:
        The name of the dva position columns in the input data frame.
    velocity_columns:
        The name of the dva velocity columns in the input data frame.
    acceleration_columns:
        The name of the dva acceleration columns in the input data frame.

    Returns
    -------
    py:class:`~pymovements.GazeDataFrame`
    """

    if data is not None:
        checks.check_is_mutual_exclusive(data=data, time=time)
        checks.check_is_mutual_exclusive(data=data, pixel=pixel)
        checks.check_is_mutual_exclusive(data=data, position=position)
        checks.check_is_mutual_exclusive(data=data, velocity=velocity)
        checks.check_is_mutual_exclusive(data=data, acceleration=acceleration)

        df = pl.from_numpy(data=data, schema=schema, orient=orient)
        return GazeDataFrame(
            data=df,
            experiment=experiment,
            time_column=time_column,
            pixel_columns=pixel_columns,
            position_columns=position_columns,
            velocity_columns=velocity_columns,
            acceleration_columns=acceleration_columns,
        )

    n_components: int = 0
    columns: pl.Series = []

    if time is not None:
        column = pl.from_numpy(data=time, schema=['time'], orient=orient)['time']
        columns.append(column)

    if pixel is not None:
        column = pl.from_numpy(data=pixel, schema=['pixel'], orient=orient)['pixel']
        columns.append(column)
        n_components = column.list.lengths()[0]

    if position is not None:
        column = pl.from_numpy(data=position, schema=['position'], orient=orient)['position']
        columns.append(column)
        n_components = column.list.lengths()[0]

    if velocity is not None:
        column = pl.from_numpy(data=velocity, schema=['velocity'], orient=orient)['velocity']
        columns.append(column)
        n_components = column.list.lengths()[0]

    if acceleration is not None:
        column = pl.from_numpy(
            data=acceleration, schema=['acceleration'], orient=orient,
        )['acceleration']
        columns.append(column)
        n_components = column.list.lengths()[0]

    df = pl.DataFrame(columns)
    gaze = GazeDataFrame(
        data=df,
        experiment=experiment,
    )
    gaze.n_components = n_components

    return gaze


def from_pandas(
        data: pd.DataFrame,
        experiment: Experiment | None = None,
        time_column: str | None = None,
        pixel_columns: list[str] | None = None,
        position_columns: list[str] | None = None,
        velocity_columns: list[str] | None = None,
        acceleration_columns: list[str] | None = None,
) -> GazeDataFrame:
    """Construct a :py:class:`~pymovements.gaze.gaze_dataframe.GazeDataFrame`
    from a pandas DataFrame.

    Parameters
    ----------
    data:
        Data represented as a pandas DataFrame.
    experiment : Experiment
        The experiment definition.
    time_column:
        The name of the timestamp column in the input data frame.
    pixel_columns:
        The name of the pixel position columns in the input data frame.
    position_columns:
        The name of the dva position columns in the input data frame.
    velocity_columns:
        The name of the dva velocity columns in the input data frame.
    acceleration_columns:
        The name of the dva acceleration columns in the input data frame.

    Returns
    -------
    py:class:`~pymovements.GazeDataFrame`
    """
    df = pl.from_pandas(data=data)
    return GazeDataFrame(
        data=df,
        experiment=experiment,
        time_column=time_column,
        pixel_columns=pixel_columns,
        position_columns=position_columns,
        velocity_columns=velocity_columns,
        acceleration_columns=acceleration_columns,
    )
