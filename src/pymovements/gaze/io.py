# Copyright (c) 2023-2024 The pymovements Project Authors
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

from pymovements.gaze.experiment import Experiment
from pymovements.gaze.eyetracker import EyeTracker
from pymovements.gaze.gaze_dataframe import GazeDataFrame  # pylint: disable=cyclic-import
from pymovements.utils.parsing import parse_eyelink


def from_csv(
        file: str | Path,
        experiment: Experiment | None = None,
        *,
        trial_columns: str | list[str] | None = None,
        time_column: str | None = None,
        time_unit: str | None = 'ms',
        pixel_columns: list[str] | None = None,
        position_columns: list[str] | None = None,
        velocity_columns: list[str] | None = None,
        acceleration_columns: list[str] | None = None,
        distance_column: str | None = None,
        column_map: dict[str, str] | None = None,
        add_columns: dict[str, str] | None = None,
        column_schema_overrides: dict[str, type] | None = None,
        **read_csv_kwargs: Any,
) -> GazeDataFrame:
    """Initialize a :py:class:`pymovements.gaze.gaze_dataframe.GazeDataFrame`.

    Parameters
    ----------
    file: str | Path
        Path of gaze file.
    experiment : Experiment | None
        The experiment definition. (default: None)
    trial_columns: str | list[str] | None
        The name of the trial columns in the input data frame. If the list is empty or None,
        the input data frame is assumed to contain only one trial. If the list is not empty,
        the input data frame is assumed to contain multiple trials and the transformation
        methods will be applied to each trial separately. (default: None)
    time_column: str | None
        The name of the timestamp column in the input data frame. (default: None)
    time_unit: str | None
        The unit of the timestamps in the timespamp column in the input data frame. Supported
        units are 's' for seconds, 'ms' for milliseconds and 'step' for steps. If the unit is
        'step' the experiment definition must be specified. All timestamps will be converted to
        milliseconds. (default: 'ms')
    pixel_columns: list[str] | None
        The name of the pixel position columns in the input data frame. These columns will be
        nested into the column ``pixel``. If the list is empty or None, the nested ``pixel``
        column will not be created. (default: None)
    position_columns: list[str] | None
        The name of the dva position columns in the input data frame. These columns will be
        nested into the column ``position``. If the list is empty or None, the nested
        ``position`` column will not be created. (default: None)
    velocity_columns: list[str] | None
        The name of the velocity columns in the input data frame. These columns will be nested
        into the column ``velocity``. If the list is empty or None, the nested ``velocity``
        column will not be created. (default: None)
    acceleration_columns: list[str] | None
        The name of the acceleration columns in the input data frame. These columns will be
        nested into the column ``acceleration``. If the list is empty or None, the nested
        ``acceleration`` column will not be created. (default: None)
    distance_column: str | None
        The name of the eye-to-screen distance column in the input data frame. If specified,
        the column will be used for pixel to dva transformations. If not specified, the
        constant eye-to-screen distance will be taken from the experiment definition.
        (default: None)
    column_map: dict[str, str] | None
        The keys are the columns to read, the values are the names to which they should be renamed.
        (default: None)
    add_columns: dict[str, str] | None
        Dictionary containing columns to add to loaded data frame.
        (default: None)
    column_schema_overrides:  dict[str, type] | None
        Dictionary containing types for columns.
        (default: None)
    **read_csv_kwargs: Any
        Additional keyword arguments to be passed to :py:func:`polars.read_csv` to read in the csv.
        These can include custom separators, a subset of columns, or specific data types
        for columns.

    Returns
    -------
    GazeDataFrame
        The gaze data frame read from the csv file.

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
    First let's assume a CSV file stored `tests/files/monocular_example.csv`
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
    │ 4    ┆ 0          ┆ 0          │
    │ 5    ┆ 0          ┆ 0          │
    │ 6    ┆ 0          ┆ 0          │
    │ 7    ┆ 0          ┆ 0          │
    │ 8    ┆ 0          ┆ 0          │
    │ 9    ┆ 0          ┆ 0          │
    └──────┴────────────┴────────────┘

    We can now load the data into a ``GazeDataFrame`` by specyfing the experimental setting
    and the names of the pixel position columns. We can specify a custom separator for the csv
    file by passing it as a keyword argument to :py:func:`polars.read_csv`:

    >>> from pymovements.gaze.io import from_csv
    >>> gaze = from_csv(
    ...     file='tests/files/monocular_example.csv',
    ...     time_column = 'time',
    ...     time_unit='ms',
    ...     pixel_columns = ['x_left_pix','y_left_pix'],
    ...     separator = ',',
    ... )
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
    │ 4    ┆ [0, 0]    │
    │ 5    ┆ [0, 0]    │
    │ 6    ┆ [0, 0]    │
    │ 7    ┆ [0, 0]    │
    │ 8    ┆ [0, 0]    │
    │ 9    ┆ [0, 0]    │
    └──────┴───────────┘

    Please be aware that data types are inferred from a fixed number of rows. To ensure
    correct data types, you can pass a dictionary of column names and data types to the
    `schema_overrides` keyword argument of :py:func:`polars.read_csv`:

    >>> from pymovements.gaze.io import from_csv
    >>> import polars as pl
    >>> gaze = from_csv(
    ...     file='tests/files/monocular_example.csv',
    ...     time_column = 'time',
    ...     time_unit='ms',
    ...     pixel_columns = ['x_left_pix','y_left_pix'],
    ...     schema_overrides = {'time': pl.Int64, 'x_left_pix': pl.Int64, 'y_left_pix': pl.Int64},
    ... )
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
    │ 4    ┆ [0, 0]    │
    │ 5    ┆ [0, 0]    │
    │ 6    ┆ [0, 0]    │
    │ 7    ┆ [0, 0]    │
    │ 8    ┆ [0, 0]    │
    │ 9    ┆ [0, 0]    │
    └──────┴───────────┘

    """
    # Read data.
    gaze_data = pl.read_csv(file, **read_csv_kwargs)
    if column_map is not None:
        gaze_data = gaze_data.rename({
            key: column_map[key] for key in
            [
                key for key in column_map.keys()
                if key in gaze_data.columns
            ]
        })

    if add_columns is not None:
        gaze_data = gaze_data.with_columns([
            pl.lit(value).alias(column)
            for column, value in add_columns.items()
            if column not in gaze_data.columns
        ])

    # Cast numerical columns to Float64 if they were incorrectly inferred to be Utf8.
    # This can happen if the column only has missing values in the top 100 rows.
    numerical_columns = (
        (pixel_columns or [])
        + (position_columns or [])
        + (velocity_columns or [])
        + (acceleration_columns or [])
        + ([distance_column] if distance_column else [])
    )
    for column in numerical_columns:
        if gaze_data[column].dtype == pl.Utf8:
            gaze_data = gaze_data.with_columns([
                pl.col(column).cast(pl.Float64),
            ])

    if column_schema_overrides is not None:
        gaze_data = gaze_data.with_columns([
            pl.col(fileinfo_key).cast(fileinfo_dtype)
            for fileinfo_key, fileinfo_dtype in column_schema_overrides.items()
        ])

    # Create gaze data frame.
    gaze_df = GazeDataFrame(
        gaze_data,
        experiment=experiment,
        trial_columns=trial_columns,
        time_column=time_column,
        time_unit=time_unit,
        pixel_columns=pixel_columns,
        position_columns=position_columns,
        velocity_columns=velocity_columns,
        acceleration_columns=acceleration_columns,
        distance_column=distance_column,
    )
    return gaze_df


def from_asc(
        file: str | Path,
        *,
        patterns: str | list[dict[str, Any] | str] | None = 'eyelink',
        metadata_patterns: list[dict[str, Any] | str] | None = None,
        schema: dict[str, Any] | None = None,
        experiment: Experiment | None = None,
        trial_columns: str | list[str] | None = None,
        add_columns: dict[str, str] | None = None,
        column_schema_overrides: dict[str, Any] | None = None,
) -> GazeDataFrame:
    """Initialize a :py:class:`pymovements.gaze.gaze_dataframe.GazeDataFrame`.

    Parameters
    ----------
    file: str | Path
        Path of IPC/feather file.
    patterns: str | list[dict[str, Any] | str] | None
        List of patterns to match for additional columns or a key identifier of eye tracker specific
        default patterns. Supported values are: eyelink. (default: 'eyelink')
    metadata_patterns: list[dict[str, Any] | str] | None
        List of patterns to match for extracting metadata from custom logged messages.
        (default: None)
    schema: dict[str, Any] | None
        Dictionary to optionally specify types of columns parsed by patterns. (default: None)
    experiment: Experiment | None
        The experiment definition. (default: None)
    trial_columns: str | list[str] | None
        The names of the columns (extracted by patterns) to use as trial columns.
        If the list is empty or None, the asc file is assumed to contain only one trial.
        If the list is not empty, the asc file is assumed to contain multiple trials and
        the transformation methods will be applied to each trial separately. (default: None)
    add_columns: dict[str, str] | None
        Dictionary containing columns to add to loaded data frame.
        (default: None)
    column_schema_overrides: dict[str, Any] | None
        Dictionary containing types for columns.
        (default: None)

    Returns
    -------
    GazeDataFrame
        The gaze data frame read from the asc file.

    Examples
    --------
    Let's assume we have an EyeLink asc file stored at `tests/files/eyelink_monocular_example.asc`.
    We can then load the data into a ``GazeDataFrame``:

    >>> from pymovements.gaze.io import from_asc
    >>> gaze = from_asc(file='tests/files/eyelink_monocular_example.asc')
    >>> gaze.frame
    shape: (16, 3)
    ┌─────────┬───────┬────────────────┐
    │ time    ┆ pupil ┆ pixel          │
    │ ---     ┆ ---   ┆ ---            │
    │ i64     ┆ f64   ┆ list[f64]      │
    ╞═════════╪═══════╪════════════════╡
    │ 2154556 ┆ 778.0 ┆ [138.1, 132.8] │
    │ 2154557 ┆ 778.0 ┆ [138.2, 132.7] │
    │ 2154560 ┆ 777.0 ┆ [137.9, 131.6] │
    │ 2154564 ┆ 778.0 ┆ [138.1, 131.0] │
    │ 2154596 ┆ 784.0 ┆ [139.6, 132.1] │
    │ …       ┆ …     ┆ …              │
    │ 2339246 ┆ 622.0 ┆ [629.9, 531.9] │
    │ 2339271 ┆ 617.0 ┆ [639.4, 531.9] │
    │ 2339272 ┆ 617.0 ┆ [639.0, 531.9] │
    │ 2339290 ┆ 618.0 ┆ [637.6, 531.4] │
    │ 2339291 ┆ 618.0 ┆ [637.3, 531.2] │
    └─────────┴───────┴────────────────┘
    >>> gaze.experiment.eyetracker.sampling_rate
    1000.0
    """
    if isinstance(patterns, str):
        if patterns == 'eyelink':
            # We use the default patterns of parse_eyelink then.
            patterns = None
        else:
            raise ValueError(f"unknown pattern key '{patterns}'. Supported keys are: eyelink")

    # Read data.
    gaze_data, metadata = parse_eyelink(
        file, patterns=patterns, schema=schema, metadata_patterns=metadata_patterns,
    )

    if add_columns is not None:
        gaze_data = gaze_data.with_columns([
            pl.lit(value).alias(column)
            for column, value in add_columns.items()
            if column not in gaze_data.columns
        ])

    if column_schema_overrides is not None:
        gaze_data = gaze_data.with_columns([
            pl.col(fileinfo_key).cast(fileinfo_dtype)
            for fileinfo_key, fileinfo_dtype in column_schema_overrides.items()
        ])

    if experiment is None:
        experiment = Experiment(sampling_rate=metadata['sampling_rate'])
    if experiment.eyetracker is None:
        experiment.eyetracker = EyeTracker()

    # Compare metadata from experiment definition with metadata from ASC file.
    # Fill in missing metadata in experiment definition and raise an error if there are conflicts
    issues = []

    # Screen resolution (assuming that width and height will always be missing or set together)
    experiment_resolution = (experiment.screen.width_px, experiment.screen.height_px)
    if experiment_resolution == (None, None):
        experiment.screen.width_px, experiment.screen.height_px = metadata['resolution']
    elif experiment_resolution != metadata['resolution']:
        issues.append(f"Screen resolution: {experiment_resolution} vs. {metadata['resolution']}")

    # Sampling rate
    if experiment.eyetracker.sampling_rate is None:
        experiment.eyetracker.sampling_rate = metadata['sampling_rate']
    elif experiment.eyetracker.sampling_rate != metadata['sampling_rate']:
        issues.append(
            f"Sampling rate: {experiment.eyetracker.sampling_rate} vs. {metadata['sampling_rate']}",
        )

    # Tracked eye
    asc_left_eye = 'L' in metadata['tracked_eye']
    asc_right_eye = 'R' in metadata['tracked_eye']
    if experiment.eyetracker.left is None:
        experiment.eyetracker.left = asc_left_eye
    elif experiment.eyetracker.left != asc_left_eye:
        issues.append(f"Left eye tracked: {experiment.eyetracker.left} vs. {asc_left_eye}")
    if experiment.eyetracker.right is None:
        experiment.eyetracker.right = asc_right_eye
    elif experiment.eyetracker.right != asc_right_eye:
        issues.append(f"Right eye tracked: {experiment.eyetracker.right} vs. {asc_right_eye}")

    # Mount configuration
    if experiment.eyetracker.mount is None:
        experiment.eyetracker.mount = metadata['mount_configuration']['mount_type']
    elif experiment.eyetracker.mount != metadata['mount_configuration']['mount_type']:
        issues.append(f"Mount configuration: {experiment.eyetracker.mount} vs. "
                      f"{metadata['mount_configuration']['mount_type']}")

    # Eye tracker vendor
    asc_vendor = 'EyeLink' if 'EyeLink' in metadata['model'] else None
    if experiment.eyetracker.vendor is None:
        experiment.eyetracker.vendor = asc_vendor
    elif experiment.eyetracker.vendor != asc_vendor:
        issues.append(f"Eye tracker vendor: {experiment.eyetracker.vendor} vs. {asc_vendor}")

    # Eye tracker model
    if experiment.eyetracker.model is None:
        experiment.eyetracker.model = metadata['model']
    elif experiment.eyetracker.model != metadata['model']:
        issues.append(f"Eye tracker model: {experiment.eyetracker.model} vs. {metadata['model']}")

    # Eye tracker software version
    if experiment.eyetracker.version is None:
        experiment.eyetracker.version = metadata['version_number']
    elif experiment.eyetracker.version != metadata['version_number']:
        issues.append(f"Eye tracker software version: {experiment.eyetracker.version} vs. "
                      f"{metadata['version_number']}")

    if issues:
        raise ValueError(
            'Experiment metadata does not match the metadata in the ASC file:\n'
            + '\n'.join(f'- {issue}' for issue in issues),
        )

    # Create gaze data frame.
    gaze_df = GazeDataFrame(
        gaze_data,
        experiment=experiment,
        trial_columns=trial_columns,
        time_column='time',
        time_unit='ms',
        pixel_columns=['x_pix', 'y_pix'],
    )
    gaze_df._metadata = metadata  # pylint: disable=protected-access
    return gaze_df


def from_ipc(
        file: str | Path,
        experiment: Experiment | None = None,
        column_map: dict[str, str] | None = None,
        add_columns: dict[str, str] | None = None,
        column_schema_overrides: dict[str, type] | None = None,
        **read_ipc_kwargs: Any,
) -> GazeDataFrame:
    """Initialize a :py:class:`pymovements.gaze.gaze_dataframe.GazeDataFrame`.

    Parameters
    ----------
    file: str | Path
        Path of IPC/feather file.
    experiment : Experiment | None
        The experiment definition.
        (default: None)
    column_map: dict[str, str] | None
        The keys are the columns to read, the values are the names to which they should be renamed.
        (default: None)
    add_columns: dict[str, str] | None
        Dictionary containing columns to add to loaded data frame.
        (default: None)
    column_schema_overrides:  dict[str, type] | None
        Dictionary containing types for columns.
        (default: None)
    **read_ipc_kwargs: Any
            Additional keyword arguments to be passed to polars to read in the ipc file.

    Returns
    -------
    GazeDataFrame
        The gaze data frame read from the ipc file.

    Examples
    --------
    Let's assume we have an IPC file stored at `tests/files/monocular_example.feather`.
    We can then load the data into a ``GazeDataFrame``:

    >>> from pymovements.gaze.io import from_ipc
    >>> gaze = from_ipc(file='tests/files/monocular_example.feather')
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
    │ 4    ┆ [0, 0]    │
    │ 5    ┆ [0, 0]    │
    │ 6    ┆ [0, 0]    │
    │ 7    ┆ [0, 0]    │
    │ 8    ┆ [0, 0]    │
    │ 9    ┆ [0, 0]    │
    └──────┴───────────┘

    """
    # Read data.
    gaze_data = pl.read_ipc(file, **read_ipc_kwargs)

    if column_map is not None:
        gaze_data = gaze_data.rename({
            key: column_map[key] for key in
            [
                key for key in column_map.keys()
                if key in gaze_data.columns
            ]
        })

    if add_columns is not None:
        gaze_data = gaze_data.with_columns([
            pl.lit(value).alias(column)
            for column, value in add_columns.items()
            if column not in gaze_data.columns
        ])

    if column_schema_overrides is not None:
        gaze_data = gaze_data.with_columns([
            pl.col(fileinfo_key).cast(fileinfo_dtype)
            for fileinfo_key, fileinfo_dtype in column_schema_overrides.items()
        ])

    # Create gaze data frame.
    gaze_df = GazeDataFrame(
        gaze_data,
        experiment=experiment,
    )
    return gaze_df
