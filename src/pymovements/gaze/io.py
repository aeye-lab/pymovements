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
"""Functionality to load GazeDataFrame from a csv file."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

from pymovements.gaze import Experiment  # pylint: disable=cyclic-import
from pymovements.gaze.gaze_dataframe import GazeDataFrame  # pylint: disable=cyclic-import


def from_csv(
        file: str | Path,
        experiment: Experiment | None = None,
        *,
        trial_columns: list[str] | None = None,
        time_column: str | None = None,
        pixel_columns: list[str] | None = None,
        position_columns: list[str] | None = None,
        velocity_columns: list[str] | None = None,
        acceleration_columns: list[str] | None = None,
        distance_column: str | None = None,
        **read_csv_kwargs: Any,
) -> GazeDataFrame:
    """Initialize a :py:class:`pymovements.gaze.gaze_dataframe.GazeDataFrame`.

    Parameters
    ----------
    file:
        Path of gaze file.
    experiment : Experiment
        The experiment definition.
    trial_columns:
        The name of the trial columns in the input data frame. If the list is empty or None,
        the input data frame is assumed to contain only one trial. If the list is not empty,
        the input data frame is assumed to contain multiple trials and the transformation
        methods will be applied to each trial separately.
    time_column:
        The name of the timestamp column in the input data frame.
    pixel_columns:
        The name of the pixel position columns in the input data frame. These columns will be
        nested into the column ``pixel``. If the list is empty or None, the nested ``pixel``
        column will not be created.
    position_columns:
        The name of the dva position columns in the input data frame. These columns will be
        nested into the column ``position``. If the list is empty or None, the nested
        ``position`` column will not be created.
    velocity_columns:
        The name of the velocity columns in the input data frame. These columns will be nested
        into the column ``velocity``. If the list is empty or None, the nested ``velocity``
        column will not be created.
    acceleration_columns:
        The name of the acceleration columns in the input data frame. These columns will be
        nested into the column ``acceleration``. If the list is empty or None, the nested
        ``acceleration`` column will not be created.
    distance_column:
        The name of the eye-to-screen distance column in the input data frame. If specified,
        the column will be used for pixel to dva transformations. If not specified, the
        constant eye-to-screen distance will be taken from the experiment definition.
    **read_csv_kwargs:
            Additional keyword arguments to be passed to polars to read in the csv.

    Notes
    -----
    About using the arguments ``pixel_columns``, ``position_columns``, ``velocity_columns``,
    and ``acceleration_columns``:

    By passing a list of columns as any of these arguments, these columns will be merged into a
    single column with the corresponding name , e.g. using `pixel_columns` will merge the
    respective columns into the column `pixel`.

    The supported number of component columns with the expected order are:

    * zero columns: No nested component column will be created.
    * two columns: monocular data; expected order: x-component, y-component
    * four columns: binocular data; expected order: x-component left eye, y-component left eye,
      x-component right eye, y-component right eye,
    * six columns: binocular data with additional cyclopian data; expected order: x-component
      left eye, y-component left eye, x-component right eye, y-component right eye,
      x-component cyclopian eye, y-component cyclopian eye,


    Examples
    --------
    First let's assume a CSV file stored `tests/gaze/io/files/monocular_example.csv`
    with the following content:
    shape: (10, 3)
    ┌──────┬────────────┬────────────┐
    │ time ┆ x_left_pix ┆ y_left_pix │
    │ ---  ┆ ---        ┆ ---        │
    │ i64  ┆ i64        ┆ i64        │
    ╞══════╪════════════╪════════════╡
    │ 0    ┆ 0          ┆ 0          │
    │ 1    ┆ 0          ┆ 0          │
    │ 2    ┆ 0          ┆ 0          │
    │ 3    ┆ 0          ┆ 0          │
    │ …    ┆ …          ┆ …          │
    │ 6    ┆ 0          ┆ 0          │
    │ 7    ┆ 0          ┆ 0          │
    │ 8    ┆ 0          ┆ 0          │
    │ 9    ┆ 0          ┆ 0          │
    └──────┴────────────┴────────────┘

    We can now load the data into a ``GazeDataFrame`` by specyfing the experimental setting
    and the names of the pixel position columns.

    >>> from pymovements.gaze.io import from_csv
    >>> gaze = from_csv(
    ...     file='tests/files/monocular_example.csv',
    ...     time_column = 'time',
    ...     pixel_columns = ['x_left_pix','y_left_pix'],)
    >>> gaze.frame
    shape: (10, 2)
    ┌──────┬───────────┐
    │ time ┆ pixel     │
    │ ---  ┆ ---       │
    │ i64  ┆ list[i64] │
    ╞══════╪═══════════╡
    │ 0    ┆ [0, 0]    │
    │ 1    ┆ [0, 0]    │
    │ 2    ┆ [0, 0]    │
    │ 3    ┆ [0, 0]    │
    │ …    ┆ …         │
    │ 6    ┆ [0, 0]    │
    │ 7    ┆ [0, 0]    │
    │ 8    ┆ [0, 0]    │
    │ 9    ┆ [0, 0]    │
    └──────┴───────────┘

    """
    # read data
    gaze_data = pl.read_csv(file, **read_csv_kwargs)

    # create gaze data frame
    gaze_df = GazeDataFrame(
        gaze_data,
        experiment=experiment,
        trial_columns=trial_columns,
        time_column=time_column,
        pixel_columns=pixel_columns,
        position_columns=position_columns,
        velocity_columns=velocity_columns,
        acceleration_columns=acceleration_columns,
        distance_column=distance_column,
    )
    return gaze_df


def from_ipc(
        file: str | Path,
        experiment: Experiment | None = None,
        **read_ipc_kwargs: Any,
) -> GazeDataFrame:
    """Initialize a :py:class:`pymovements.gaze.gaze_dataframe.GazeDataFrame`.

    Parameters
    ----------
    file:
        Path of IPC/feather file.
    experiment : Experiment
        The experiment definition.
    **read_ipc_kwargs:
            Additional keyword arguments to be passed to polars to read in the ipc file.

    Examples
    --------
    Let's assume we have an IPC file stored at `tests/gaze/io/files/monocular_example.feather`.
    We can then load the data into a ``GazeDataFrame``:

    >>> from pymovements.gaze.io import from_ipc
    >>> gaze = from_ipc(
    ...     file='tests/files/monocular_example.feather',
    ...     )
    >>> gaze.frame
    shape: (10, 2)
    ┌──────┬───────────┐
    │ time ┆ pixel     │
    │ ---  ┆ ---       │
    │ i64  ┆ list[i64] │
    ╞══════╪═══════════╡
    │ 0    ┆ [0, 0]    │
    │ 1    ┆ [0, 0]    │
    │ 2    ┆ [0, 0]    │
    │ 3    ┆ [0, 0]    │
    │ …    ┆ …         │
    │ 6    ┆ [0, 0]    │
    │ 7    ┆ [0, 0]    │
    │ 8    ┆ [0, 0]    │
    │ 9    ┆ [0, 0]    │
    └──────┴───────────┘

    """
    # read data
    gaze_data = pl.read_ipc(file, **read_ipc_kwargs)

    # create gaze data frame
    gaze_df = GazeDataFrame(
        gaze_data,
        experiment=experiment,
    )
    return gaze_df
