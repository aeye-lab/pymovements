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
"""Module for the GazeDataFrame."""
from __future__ import annotations

import inspect
import warnings
from collections.abc import Callable
from copy import deepcopy
from typing import Any

import numpy as np
import polars as pl

import pymovements as pm  # pylint: disable=cyclic-import
from pymovements.gaze import transforms
from pymovements.gaze.experiment import Experiment
from pymovements.utils import checks


class GazeDataFrame:
    """A DataFrame for gaze time series data.

    Each row is a sample at a specific timestep.
    Each column is a channel in the gaze time series.

    Parameters
    ----------
    data: pl.DataFrame | None
        A dataframe to be transformed to a polars dataframe. (default: None)
    experiment : Experiment | None
        The experiment definition. (default: None)
    events: pm.EventDataFrame | None
        A dataframe of events in the gaze signal. (default: None)
    trial_columns: str | list[str] | None
        The name of the trial columns in the input data frame. If the list is empty or None,
        the input data frame is assumed to contain only one trial. If the list is not empty,
        the input data frame is assumed to contain multiple trials and the transformation
        methods will be applied to each trial separately. (default: None)
    time_column: str | None
        The name of the timestamp column in the input data frame. This column will be renamed to
        ``time``. (default: None)
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
        The name of the column containing eye-to-screen distance in millimeters for each sample
        in the input data frame. If specified, the column will be used for pixel to dva
        transformations. If not specified, the constant eye-to-screen distance will be taken
        from the experiment definition. This column will be renamed to ``distance``. (default: None)

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
    First let's create an example `DataFrame` with three columns:
    the timestamp ``t`` and ``x`` and ``y`` for the pixel position.

    >>> df = pl.from_dict(
    ...     data={'t': [1000, 1001, 1002], 'x': [0.1, 0.2, 0.3], 'y': [0.1, 0.2, 0.3]},
    ... )
    >>> df
    shape: (3, 3)
    ┌──────┬─────┬─────┐
    │ t    ┆ x   ┆ y   │
    │ ---  ┆ --- ┆ --- │
    │ i64  ┆ f64 ┆ f64 │
    ╞══════╪═════╪═════╡
    │ 1000 ┆ 0.1 ┆ 0.1 │
    │ 1001 ┆ 0.2 ┆ 0.2 │
    │ 1002 ┆ 0.3 ┆ 0.3 │
    └──────┴─────┴─────┘

    We can now initialize our ``GazeDataFrame`` by specyfing the names of the time and pixel
    position columns.

    >>> gaze = GazeDataFrame(data=df, time_column='t', pixel_columns=['x', 'y'])
    >>> gaze.frame
    shape: (3, 2)
    ┌──────┬────────────┐
    │ time ┆ pixel      │
    │ ---  ┆ ---        │
    │ i64  ┆ list[f64]  │
    ╞══════╪════════════╡
    │ 1000 ┆ [0.1, 0.1] │
    │ 1001 ┆ [0.2, 0.2] │
    │ 1002 ┆ [0.3, 0.3] │
    └──────┴────────────┘

    In case your data has no time column available, you can pass an :py:class:``Experiment`` to
    create a time column with the correct sampling rate during initialization. The time column will
    be represented in millisecond units.

    >>> df_no_time = df.select(pl.exclude('t'))
    >>> df_no_time
    shape: (3, 2)
    ┌─────┬─────┐
    │ x   ┆ y   │
    │ --- ┆ --- │
    │ f64 ┆ f64 │
    ╞═════╪═════╡
    │ 0.1 ┆ 0.1 │
    │ 0.2 ┆ 0.2 │
    │ 0.3 ┆ 0.3 │
    └─────┴─────┘

    >>> experiment = Experiment(1024, 768, 38, 30, 60, 'center', sampling_rate=100)
    >>> gaze = GazeDataFrame(data=df_no_time, experiment=experiment, pixel_columns=['x', 'y'])
    >>> gaze.frame
    shape: (3, 2)
    ┌──────┬────────────┐
    │ time ┆ pixel      │
    │ ---  ┆ ---        │
    │ f64  ┆ list[f64]  │
    ╞══════╪════════════╡
    │ 0.0  ┆ [0.1, 0.1] │
    │ 10.0 ┆ [0.2, 0.2] │
    │ 20.0 ┆ [0.3, 0.3] │
    └──────┴────────────┘

    Leaving out the experiment definition will create a continuous integer column in step units.
    >>> gaze = GazeDataFrame(data=df_no_time, pixel_columns=['x', 'y'])
    >>> gaze.frame
    shape: (3, 2)
    ┌──────┬────────────┐
    │ time ┆ pixel      │
    │ ---  ┆ ---        │
    │ i64  ┆ list[f64]  │
    ╞══════╪════════════╡
    │ 0    ┆ [0.1, 0.1] │
    │ 1    ┆ [0.2, 0.2] │
    │ 2    ┆ [0.3, 0.3] │
    └──────┴────────────┘

    """

    valid_pixel_position_columns = [
        'x_pix', 'y_pix',
        'x_left_pix', 'y_left_pix',
        'x_right_pix', 'y_right_pix',
        '__x_pix__', '__y_pix__',
        '__x_left_pix__', '__y_left_pix__',
        '__x_right_pix__', '__y_right_pix__',
    ]

    valid_position_columns = [
        'x_pos', 'y_pos',
        'x_left_pos', 'y_left_pos',
        'x_right_pos', 'y_right_pos',
        '__x_pos__', '__y_pos__',
        '__x_left_pos__', '__y_left_pos__',
        '__x_right_pos__', '__y_right_pos__',
    ]

    valid_velocity_columns = [
        'x_vel', 'y_vel',
        'x_left_vel', 'y_left_vel',
        'x_right_vel', 'y_right_vel',
        '__x_vel__', '__y_vel__',
        '__x_left_vel__', '__y_left_vel__',
        '__x_right_vel__', '__y_right_vel__',
    ]

    valid_acceleration_columns = [
        'x_acc', 'y_acc',
        'x_left_acc', 'y_left_acc',
        'x_right_acc', 'y_right_acc',
        '__x_acc__', '__y_acc__',
        '__x_left_acc__', '__y_left_acc__',
        '__x_right_acc__', '__y_right_acc__',
    ]

    def __init__(
            self,
            data: pl.DataFrame | None = None,
            experiment: Experiment | None = None,
            events: pm.EventDataFrame | None = None,
            *,
            trial_columns: str | list[str] | None = None,
            time_column: str | None = None,
            pixel_columns: list[str] | None = None,
            position_columns: list[str] | None = None,
            velocity_columns: list[str] | None = None,
            acceleration_columns: list[str] | None = None,
            distance_column: str | None = None,
    ):
        if data is None:
            data = pl.DataFrame()
        else:
            data = data.clone()
        self.frame = data

        self.trial_columns = [trial_columns] if isinstance(trial_columns, str) else trial_columns

        # In case the 'time' column is already present we don't need to do anything.
        # Otherwise, create a new time column starting with zero.
        if time_column is None and 'time' not in self.frame.columns:
            # In case we have an experiment with sampling rate given, we convert to milliseconds.
            if experiment is not None and experiment.sampling_rate is not None:
                sampling_rate_factor = 1000 / experiment.sampling_rate
            else:
                sampling_rate_factor = 1

            self.frame = self.frame.with_columns(
                time=pl.arange(0, len(self.frame)) * sampling_rate_factor
            )

        # This if clause is mutually exclusive with the previous one.
        if time_column is not None:
            self.frame = self.frame.rename({time_column: 'time'})

        if distance_column is not None:
            self.frame = self.frame.rename({distance_column: 'distance'})

        # List of passed not-None column specifier lists.
        # The list will be used for inferring n_components.
        column_specifiers: list[list[str]] = []

        if pixel_columns:
            self._check_component_columns(pixel_columns=pixel_columns)
            self.nest(pixel_columns, output_column='pixel')
            column_specifiers.append(pixel_columns)

        if position_columns:
            self._check_component_columns(position_columns=position_columns)
            self.nest(position_columns, output_column='position')
            column_specifiers.append(position_columns)

        if velocity_columns:
            self._check_component_columns(velocity_columns=velocity_columns)
            self.nest(velocity_columns, output_column='velocity')
            column_specifiers.append(velocity_columns)

        if acceleration_columns:
            self._check_component_columns(acceleration_columns=acceleration_columns)
            self.nest(acceleration_columns, output_column='acceleration')
            column_specifiers.append(acceleration_columns)

        self.n_components = self._infer_n_components(column_specifiers)
        self.experiment = experiment

        if events is None:
            if self.trial_columns is None:
                self.events = pm.EventDataFrame()
            else:  # Ensure that trial columns with correct dtype are present in event dataframe.
                self.events = pm.EventDataFrame(
                    data=pl.DataFrame(
                        schema={column: self.frame.schema[column] for column in self.trial_columns},
                    )
                )
        else:
            self.events = events.copy()

    def apply(
            self,
            function: str,
            **kwargs: Any,
    ) -> None:
        """Apply preprocessing method to GazeDataFrame.

        Parameters
        ----------
        function: str
            Name of the preprocessing function to apply.
        **kwargs: Any
            kwargs that will be forwarded when calling the preprocessing method.
        """
        if transforms.TransformLibrary.__contains__(function):
            self.transform(function, **kwargs)
        elif pm.events.EventDetectionLibrary.__contains__(function):
            self.detect(function, **kwargs)
        else:
            raise ValueError(f"unsupported method '{function}'")

    def transform(
            self,
            transform_method: str | Callable[..., pl.Expr],
            **kwargs: Any,
    ) -> None:
        """Apply transformation method.

        Parameters
        ----------
        transform_method: str | Callable[..., pl.Expr]
            The transformation method to be applied.
        **kwargs: Any
            Additional keyword arguments to be passed to the transformation method.
        """
        if isinstance(transform_method, str):
            transform_method = transforms.TransformLibrary.get(transform_method)

        if transform_method.__name__ == 'downsample':
            downsample_factor = kwargs.pop('factor')
            self.frame = self.frame.select(
                transforms.downsample(
                    factor=downsample_factor, **kwargs,
                ),
            )

        else:
            method_kwargs = inspect.getfullargspec(transform_method).kwonlyargs
            if 'origin' in method_kwargs and 'origin' not in kwargs:
                self._check_experiment()
                assert self.experiment is not None
                kwargs['origin'] = self.experiment.screen.origin

            if 'screen_resolution' in method_kwargs and 'screen_resolution' not in kwargs:
                self._check_experiment()
                assert self.experiment is not None
                kwargs['screen_resolution'] = (
                    self.experiment.screen.width_px, self.experiment.screen.height_px,
                )

            if 'screen_size' in method_kwargs and 'screen_size' not in kwargs:
                self._check_experiment()
                assert self.experiment is not None
                kwargs['screen_size'] = (
                    self.experiment.screen.width_cm, self.experiment.screen.height_cm,
                )

            if 'distance' in method_kwargs and 'distance' not in kwargs:
                self._check_experiment()
                assert self.experiment is not None

                if 'distance' in self.frame.columns:
                    kwargs['distance'] = 'distance'

                    if self.experiment.screen.distance_cm:
                        warnings.warn(
                            "Both a distance column and experiment's "
                            'eye-to-screen distance are specified. '
                            'Using eye-to-screen distances from column '
                            "'distance' in the dataframe.",
                        )
                elif self.experiment.screen.distance_cm:
                    kwargs['distance'] = self.experiment.screen.distance_cm
                else:
                    raise AttributeError(
                        'Neither eye-to-screen distance is in the columns of the dataframe '
                        'nor experiment eye-to-screen distance is specified.',
                    )

            if 'sampling_rate' in method_kwargs and 'sampling_rate' not in kwargs:
                self._check_experiment()
                assert self.experiment is not None
                kwargs['sampling_rate'] = self.experiment.sampling_rate

            if 'n_components' in method_kwargs and 'n_components' not in kwargs:
                self._check_n_components()
                kwargs['n_components'] = self.n_components

            if transform_method.__name__ in {'pos2vel', 'pos2acc'}:
                if 'position' not in self.frame.columns and 'position_column' not in kwargs:
                    if 'pixel' in self.frame.columns:
                        raise pl.exceptions.ColumnNotFoundError(
                            "Neither is 'position' in the dataframe columns, "
                            'nor is a position column explicitly specified. '
                            "Since the dataframe has a 'pixel' column, consider running "
                            f'pix2deg() before {transform_method.__name__}(). If you want '
                            'to run transformations in pixel units, you can do so by using '
                            f"{transform_method.__name__}(position_column='pixel'). "
                            f'Available dataframe columns are: {self.frame.columns}',
                        )
                    raise pl.exceptions.ColumnNotFoundError(
                        "Neither is 'position' in the dataframe columns, "
                        'nor is a position column explicitly specified. '
                        'You can specify the position column via: '
                        f'{transform_method.__name__}(position_column="your_position_column"). '
                        f'Available dataframe columns are: {self.frame.columns}',
                    )
            if transform_method.__name__ in {'pix2deg'}:
                if 'pixel' not in self.frame.columns and 'pixel_column' not in kwargs:
                    raise pl.exceptions.ColumnNotFoundError(
                        "Neither is 'pixel' in the dataframe columns, "
                        'nor is a pixel column explicitly specified. '
                        'You can specify the pixel column via: '
                        f'{transform_method.__name__}(pixel_column="name_of_your_pixel_column"). '
                        f'Available dataframe columns are: {self.frame.columns}',
                    )

            if self.trial_columns is None:
                self.frame = self.frame.with_columns(transform_method(**kwargs))
            else:
                self.frame = pl.concat(
                    [
                        df.with_columns(transform_method(**kwargs))
                        for group, df in
                        self.frame.group_by(self.trial_columns, maintain_order=True)
                    ],
                )

    def pix2deg(self) -> None:
        """Compute gaze positions in degrees of visual angle from pixel position coordinates.

        This method requires a properly initialized :py:attr:`~.GazeDataFrame.experiment` attribute.

        After success, the gaze dataframe is extended by the resulting dva position columns.

        Raises
        ------
        AttributeError
            If `gaze` is None or there are no gaze dataframes present in the `gaze` attribute, or
            if experiment is None.
        """
        self.transform('pix2deg')

    def pos2acc(
            self,
            *,
            degree: int = 2,
            window_length: int = 7,
            padding: str | float | int | None = 'nearest',
    ) -> None:
        """Compute gaze acceleration in dva/s^2 from dva position coordinates.

        This method requires a properly initialized :py:attr:`~.GazeDataFrame.experiment` attribute.

        After success, the gaze dataframe is extended by the resulting velocity columns.

        Parameters
        ----------
        degree: int
            The degree of the polynomial to use. (default: 2)
        window_length: int
            The window size to use. (default: 7)
        padding: str | float | int | None
            The padding method to use. See ``savitzky_golay`` for details. (default: 'nearest')

        Raises
        ------
        AttributeError
            If `gaze` is None or there are no gaze dataframes present in the `gaze` attribute, or
            if experiment is None.
        """
        self.transform('pos2acc', window_length=window_length, degree=degree, padding=padding)

    def pos2vel(
            self,
            method: str = 'fivepoint',
            **kwargs: int | float | str,
    ) -> None:
        """Compute gaze velocity in dva/s from dva position coordinates.

        This method requires a properly initialized :py:attr:`~.GazeDataFrame.experiment` attribute.

        After success, the gaze dataframe is extended by the resulting velocity columns.

        Parameters
        ----------
        method: str
            Computation method. See :func:`~transforms.pos2vel()` for details, default: fivepoint.
            (default: 'fivepoint')
        **kwargs: int | float | str
            Additional keyword arguments to be passed to the :func:`~transforms.pos2vel()` method.

        Raises
        ------
        AttributeError
            If `gaze` is None or there are no gaze dataframes present in the `gaze` attribute, or
            if experiment is None.
        """
        self.transform('pos2vel', method=method, **kwargs)

    def smooth(
            self,
            method: str = 'savitzky_golay',
            window_length: int = 7,
            degree: int = 2,
            column: str = 'position',
            padding: str | float | int | None = 'nearest',
            **kwargs: int | float | str,
    ) -> None:
        """Smooth data in a column.

        Parameters
        ----------
        method: str
            The method to use for smoothing. Choose from ``savitzky_golay``, ``moving_average``,
            ``exponential_moving_average``. See :func:`~transforms.smooth()` for details.
            (default: 'savitzky_golay')
        window_length: int
            For ``moving_average`` this is the window size to calculate the mean of the subsequent
            samples. For ``savitzky_golay`` this is the window size to use for the polynomial fit.
            For ``exponential_moving_average`` this is the span parameter. (default: 7)
        degree: int
            The degree of the polynomial to use. This has only an effect if using
            ``savitzky_golay`` as smoothing method. `degree` must be less than `window_length`.
            (default: 2)
        column: str
            The input column name to which the smoothing is applied. (default: 'position')
        padding: str | float | int | None
            Must be either ``None``, a scalar or one of the strings
            ``mirror``, ``nearest`` or ``wrap``.
            This determines the type of extension to use for the padded signal to
            which the filter is applied.
            When passing ``None``, no extension padding is used.
            When passing a scalar value, data will be padded using the passed value.
            See :func:`~transforms.smooth()` for details on the padding methods.
            (default: 'nearest')
        **kwargs: int | float | str
            Additional keyword arguments to be passed to the :func:`~transforms.smooth()` method.
        """
        self.transform(
            'smooth',
            column=column,
            method=method,
            degree=degree,
            window_length=window_length,
            padding=padding,
            **kwargs,
        )

    def detect(
            self,
            method: Callable[..., pm.EventDataFrame] | str,
            *,
            eye: str = 'auto',
            clear: bool = False,
            **kwargs: Any,
    ) -> None:
        """Detect events by applying a specific event detection method.

        Parameters
        ----------
        method: Callable[..., pm.EventDataFrame] | str
            The event detection method to be applied.
        eye: str
            Select which eye to choose. Valid options are ``auto``, ``left``, ``right`` or ``None``.
            If ``auto`` is passed, eye is inferred in the order ``['right', 'left', 'eye']`` from
            the available :py:attr:`~.Dataset.gaze` dataframe columns. (default: 'auto')
        clear: bool
            If ``True``, event DataFrame will be overwritten with new DataFrame instead of being
            merged into the existing one. (default: False)
        **kwargs: Any
            Additional keyword arguments to be passed to the event detection method.
        """
        if not self.events or clear:
            if self.trial_columns is None:
                self.events = pm.EventDataFrame()
            else:  # Ensure that trial columns with correct dtype are present in event dataframe.
                self.events = pm.EventDataFrame(
                    data=pl.DataFrame(
                        schema={column: self.frame.schema[column] for column in self.trial_columns},
                    )
                )

        if isinstance(method, str):
            method = pm.events.EventDetectionLibrary.get(method)

        if self.n_components is not None:
            eye_components = self._infer_eye_components(eye)
        else:
            eye_components = None

        if self.trial_columns is None:
            method_kwargs = self._fill_event_detection_kwargs(
                method,
                gaze=self.frame,
                events=self.events,
                eye_components=eye_components,
                **kwargs,
            )

            new_events = method(**method_kwargs)

            self.events.frame = pl.concat(
                [self.events.frame, new_events.frame],
                how='diagonal',
            )
        else:
            grouped_frames = self.frame.partition_by(
                self.trial_columns, maintain_order=True, include_key=True, as_dict=True,
            )

            new_events_grouped: list[pl.DataFrame] = []

            for group_identifier, group_gaze in grouped_frames.items():
                missing_trial_columns = [
                    trial_column for trial_column in self.trial_columns
                    if trial_column not in self.events.frame.columns
                ]
                if missing_trial_columns:
                    raise RuntimeError(
                        f'trial columns {missing_trial_columns} missing from events, '
                        f'available columns: {self.events.frame.columns}'
                    )

                method_kwargs = self._fill_event_detection_kwargs(
                    method,
                    gaze=group_gaze,
                    events=self.events,  # FIXME: use group events
                    eye_components=eye_components,
                    **kwargs,
                )

                new_events = method(**method_kwargs)
                # add group identifiers as new columns
                new_events.add_trial_column(self.trial_columns, group_identifier)

                new_events_grouped.append(new_events.frame)

            self.events.frame = pl.concat(
                [self.events.frame, *new_events_grouped],
                how='diagonal',
            )
            # ensure trial columns are at the start
            self.events.frame = self.events.frame.select(
                [
                    *self.trial_columns,
                    *[
                        column for column in self.events.frame.columns
                        if column not in self.trial_columns
                    ],
                ]
            )

    @property
    def schema(self) -> pl.type_aliases.SchemaDict:
        """Schema of event dataframe."""
        return self.frame.schema

    @property
    def columns(self) -> list[str]:
        """List of column names."""
        return self.frame.columns

    def nest(
            self,
            input_columns: list[str],
            output_column: str,
    ) -> None:
        """Nest component columns into a single tuple column.

        Input component columns will be dropped.

        Parameters
        ----------
        input_columns: list[str]
            Names of input columns to be merged into a single tuple column.
        output_column: str
            Name of the resulting tuple column.
        """
        self._check_component_columns(**{output_column: input_columns})

        self.frame = self.frame.with_columns(
            pl.concat_list([pl.col(component) for component in input_columns])
            .alias(output_column),
        ).drop(input_columns)

    def unnest(
            self,
            input_columns: list[str] | str | None = None,
            output_suffixes: list[str] | None = None,
            *,
            output_columns: list[str] | None = None,
    ) -> None:
        """Explode a column of type ``pl.List`` into one column for each list component.

        The input column will be dropped.

        Parameters
        ----------
        input_columns: list[str] | str | None
            Name(s) of input column(s) to be unnested into several component columns.
            If None all list columns 'pixel', 'position', 'velocity' and
            'acceleration' will be unnested if existing. (default: None)
        output_suffixes: list[str] | None
            Suffixes to append to the column names. (default: None)
        output_columns: list[str] | None
            Name of the resulting tuple columns. (default: None)

        Raises
        ------
        ValueError
            If both output_columns and output_suffixes are specified.
            If number of output columns / suffixes does not match number of components.
            If output columns / suffixes are not unique.
            If no columns to unnest exist and none are specified.
            If output columns are specified and more than one input column is specified.
        AttributeError
            If number of components is not 2, 4 or 6.
        Warning
            If no columns to unnest exist and none are specified.
        """
        if input_columns is None:
            cols = ['pixel', 'position', 'velocity', 'acceleration']
            input_columns = [col for col in cols if col in self.frame.columns]

            if len(input_columns) == 0:
                raise Warning(
                    'No columns to unnest. '
                    'Please specify columns to unnest via the "input_columns" argument.',
                )

        if isinstance(input_columns, str):
            input_columns = [input_columns]

        # no support for custom output columns if more than one input column will be unnested
        if output_columns is not None and not len(input_columns) == 1:
            raise ValueError(
                'You cannot specify output columns if you want to unnest more than '
                'one input column. Please specify output suffixes or use a single '
                'input column instead.',
            )

        checks.check_is_mutual_exclusive(
            output_columns=output_columns,
            output_suffixes=output_suffixes,
        )

        self._check_n_components()
        assert self.n_components in {2, 4, 6}

        col_names = [output_columns] if output_columns is not None else []

        if output_columns is None and output_suffixes is None:
            if self.n_components == 2:
                output_suffixes = ['_x', '_y']
            elif self.n_components == 4:
                output_suffixes = ['_xl', '_yl', '_xr', '_yr']
            else:  # This must be 6 as we already have checked our n_components.
                output_suffixes = ['_xl', '_yl', '_xr', '_yr', '_xa', '_ya']

        if output_suffixes:
            col_names = [
                [f'{input_col}{suffix}' for suffix in output_suffixes]
                for input_col in input_columns
            ]

        if len([
            name for name_list in col_names for name in name_list
        ]) != self.n_components * len(input_columns):
            raise ValueError(
                f'Number of output columns / suffixes ({len(col_names[0])}) '
                f'must match number of components ({self.n_components})',
            )

        if len({name for name_list in col_names for name in name_list}) != len(
                [name for name_list in col_names for name in name_list],
        ):
            raise ValueError('Output columns / suffixes must be unique')

        for input_col, column_names in zip(input_columns, col_names):
            self.frame = self.frame.with_columns(
                [
                    pl.col(input_col).list.get(component_id).alias(names)
                    for component_id, names in enumerate(column_names)
                ],
            ).drop(input_col)

    def clone(self) -> GazeDataFrame:
        """Return a copy of the GazeDataFrame.

        Returns
        -------
        GazeDataFrame
            A copy of the GazeDataFrame.
        """
        gaze = GazeDataFrame(
            data=self.frame.clone(),
            experiment=deepcopy(self.experiment),
        )
        gaze.n_components = self.n_components
        return gaze

    def _check_experiment(self) -> None:
        """Check if experiment attribute has been set.

        Raises
        ------
        AttributeError
            If experiment is None.
        """
        if self.experiment is None:
            raise AttributeError('experiment must not be None for this method to work')

    def _check_n_components(self) -> None:
        """Check that n_components is either 2, 4 or 6.

        Raises
        ------
        AttributeError
            If n_components is not 2, 4 or 6.
        """
        if self.n_components not in {2, 4, 6}:
            raise AttributeError(
                f'n_components must be either 2, 4 or 6 but is {self.n_components}',
            )

    def _check_component_columns(self, **kwargs: list[str]) -> None:
        """Check if component columns are in valid format.

        Parameters
        ----------
        **kwargs: list[str]
            Keyword arguments of component columns.
        """
        for component_type, columns in kwargs.items():
            if not isinstance(columns, list):
                raise TypeError(
                    f'{component_type} must be of type list, '
                    f'but is of type {type(columns).__name__}',
                )

            for column in columns:
                if not isinstance(column, str):
                    raise TypeError(
                        f'all elements in {component_type} must be of type str, '
                        f'but one of the elements is of type {type(column).__name__}',
                    )

            if len(columns) not in [2, 4, 6]:
                raise ValueError(
                    f'{component_type} must contain either 2, 4 or 6 columns, '
                    f'but has {len(columns)}',
                )

            for column in columns:
                if column not in self.frame.columns:
                    raise pl.exceptions.ColumnNotFoundError(
                        f'column {column} from {component_type} is not available in dataframe',
                    )

            if len(set(self.frame[columns].dtypes)) != 1:
                types_list = sorted([str(t) for t in set(self.frame[columns].dtypes)])
                raise ValueError(
                    f'all columns in {component_type} must be of same type, '
                    f'but types are {types_list}',
                )

    def _infer_n_components(self, column_specifiers: list[list[str]]) -> int | None:
        """Infer number of components from DataFrame.

        Method checks nested columns `pixel`, `position`, `velocity` and `acceleration` for number
        of components by getting their list lenghts, which must be equal for all else a ValueError
        is raised. Additionally, a list of list of column specifiers is checked for consistency.

        Parameters
        ----------
        column_specifiers: list[list[str]]
            List of list of column specifiers.

        Returns
        -------
        int | None
            Number of components

        Raises
        ------
        ValueError
            If number of components is not equal for all considered columns and rows.
        """
        all_considered_columns = ['pixel', 'position', 'velocity', 'acceleration']
        considered_columns = [
            column for column in all_considered_columns if column in self.frame.columns
        ]

        list_lengths = {
            list_length
            for column in considered_columns
            for list_length in self.frame.get_column(column).list.len().unique().to_list()
        }

        for column_specifier_list in column_specifiers:
            list_lengths.add(len(column_specifier_list))

        if len(list_lengths) > 1:
            raise ValueError(f'inconsistent number of components inferred: {list_lengths}')

        if len(list_lengths) == 0:
            return None

        return next(iter(list_lengths))

    def _infer_eye_components(self, eye: str) -> tuple[int, int]:
        """Infer eye components from eye string.

        Parameters
        ----------
        eye: str
            String specificer for inferring eye components. Supported values are: auto, mono, left
            right, cyclops. Default: auto.

        Returns
        -------
        tuple[int, int]
            Tuple of eye component indices.
        """
        self._check_n_components()

        if eye == 'auto':
            # Order of inference: cyclops, right, left.
            if self.n_components == 6:
                eye_components = 4, 5
            elif self.n_components == 4:
                eye_components = 2, 3
            else:  # We already checked number of components, must be 2.
                eye_components = 0, 1
        elif eye == 'left':
            if isinstance(self.n_components, int) and self.n_components < 4:
                # Left only makes sense if there are at least two eyes.
                raise AttributeError(
                    'left eye is only supported for data with at least 4 components',
                )
            eye_components = 0, 1
        elif eye == 'right':
            if isinstance(self.n_components, int) and self.n_components < 4:
                # Right only makes sense if there are at least two eyes.
                raise AttributeError(
                    'right eye is only supported for data with at least 4 components',
                )
            eye_components = 2, 3
        elif eye == 'cyclops':
            if isinstance(self.n_components, int) and self.n_components < 6:
                raise AttributeError(
                    'cyclops eye is only supported for data with at least 6 components',
                )
            eye_components = 4, 5
        else:
            raise ValueError(
                f"unknown eye '{eye}'. Supported values are: ['auto', 'left', 'right', 'cyclops']",
            )

        return eye_components

    def _fill_event_detection_kwargs(
            self,
            method: Callable[..., pm.EventDataFrame],
            gaze: pl.DataFrame,
            events: pm.EventDataFrame,
            eye_components: tuple[int, int] | None,
            **kwargs: Any,
    ) -> dict[str, Any]:
        """Fill event detection method kwargs with gaze attributes.

        Parameters
        ----------
        method: Callable[..., pm.EventDataFrame]
            The method for which the keyword argument dictionary will be filled.
        gaze: pl.DataFrame
            The gaze dataframe to be used for filling event detection keyword arguments.
        events: pm.EventDataFrame
            The event dataframe to be used for filling event detection keyword arguments.
        eye_components: tuple[int, int] | None
            The eye components to be used for filling event detection keyword arguments.
        **kwargs: Any
            The source keyword arguments passed to the `GazeDataFrame.detect()` method.

        Returns
        -------
        dict[str, Any]
            The filled keyword argument dictionary.
        """
        # Automatically infer eye to use for event detection.
        method_args = inspect.getfullargspec(method).args

        if 'positions' in method_args:
            if 'position' not in gaze.columns:
                raise pl.exceptions.ColumnNotFoundError(
                    f'Column \'position\' not found.'
                    f' Available columns are: {gaze.columns}',
                )

            if eye_components is None:
                raise ValueError(
                    'eye_components must not be None if passing position to event detection',
                )

            kwargs['positions'] = np.vstack(
                [
                    gaze.get_column('position').list.get(eye_component)
                    for eye_component in eye_components
                ],
            ).transpose()

        if 'velocities' in method_args:
            if 'velocity' not in gaze.columns:
                raise pl.exceptions.ColumnNotFoundError(
                    f'Column \'velocity\' not found.'
                    f' Available columns are: {gaze.columns}',
                )

            if eye_components is None:
                raise ValueError(
                    'eye_components must not be None if passing velocity to event detection',
                )

            kwargs['velocities'] = np.vstack(
                [
                    gaze.get_column('velocity').list.get(eye_component)
                    for eye_component in eye_components
                ],
            ).transpose()

        if 'events' in method_args:
            kwargs['events'] = events

        if 'timesteps' in method_args and 'time' in gaze.columns:
            kwargs['timesteps'] = gaze.get_column('time').to_numpy()

        return kwargs
