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
"""Module for the GazeDataFrame."""
from __future__ import annotations

import inspect
from collections.abc import Callable
from copy import deepcopy
from typing import Any

import polars as pl

import pymovements as pm
from pymovements.gaze import transforms
from pymovements.gaze.experiment import Experiment
from pymovements.utils import checks


class GazeDataFrame:
    """A DataFrame for gaze time series data.

    Each row is a sample at a specific timestep.
    Each column is a channel in the gaze time series.
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
    ):
        """Initialize a :py:class:`pymovements.gaze.gaze_dataframe.GazeDataFrame`.

        Parameters
        ----------
        data: pl.DataFrame
            A dataframe to be transformed to a polars dataframe.
        experiment : Experiment
            The experiment definition.
        events: EventDataFrame
            A dataframe of events in the gaze signal.
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

        We can now initialize our ``GazeDataFrame`` by specyfing the names of the pixel position
        columns.

        >>> gaze = GazeDataFrame(data=df, pixel_columns=['x', 'y'])
        >>> gaze.frame
        shape: (3, 2)
        ┌──────┬────────────┐
        │ t    ┆ pixel      │
        │ ---  ┆ ---        │
        │ i64  ┆ list[f64]  │
        ╞══════╪════════════╡
        │ 1000 ┆ [0.1, 0.1] │
        │ 1001 ┆ [0.2, 0.2] │
        │ 1002 ┆ [0.3, 0.3] │
        └──────┴────────────┘

        """
        if data is None:
            data = pl.DataFrame()
        else:
            data = data.clone()
        self.frame = data

        self.trial_columns = trial_columns

        if time_column is not None:
            self.frame = self.frame.rename({time_column: 'time'})

        # List of passed not-None column specifier lists.
        # The list will be used for inferring n_components.
        column_specifiers: list[list[str]] = []

        if pixel_columns:
            _check_component_columns(self.frame, pixel_columns=pixel_columns)
            self.nest(pixel_columns, output_column='pixel')
            column_specifiers.append(pixel_columns)

        if position_columns:
            _check_component_columns(self.frame, position_columns=position_columns)
            self.nest(position_columns, output_column='position')
            column_specifiers.append(position_columns)

        if velocity_columns:
            _check_component_columns(self.frame, velocity_columns=velocity_columns)
            self.nest(velocity_columns, output_column='velocity')
            column_specifiers.append(velocity_columns)

        if acceleration_columns:
            _check_component_columns(self.frame, acceleration_columns=acceleration_columns)
            self.nest(acceleration_columns, output_column='acceleration')
            column_specifiers.append(acceleration_columns)

        self.n_components = _infer_n_components(self.frame, column_specifiers)
        self.experiment = experiment

        if events is None:
            self.events = pm.EventDataFrame()
        else:
            self.events = events.copy()

    def transform(
            self,
            transform_method: str | Callable[..., pl.Expr],
            **kwargs: Any,
    ) -> None:
        """Apply transformation method."""
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
                kwargs['distance'] = self.experiment.screen.distance_cm

            if 'sampling_rate' in method_kwargs and 'sampling_rate' not in kwargs:
                self._check_experiment()
                assert self.experiment is not None
                kwargs['sampling_rate'] = self.experiment.sampling_rate

            if 'n_components' in method_kwargs and 'n_components' not in kwargs:
                _check_n_components(self.n_components)
                kwargs['n_components'] = self.n_components

            if transform_method.__name__ in {'pos2vel', 'pos2acc'}:
                if 'position' not in self.frame.columns and 'position_column' not in kwargs:
                    if 'pixel' in self.frame.columns:
                        raise pl.exceptions.ColumnNotFoundError(
                            "Neither 'position' is in the columns of the dataframe: "
                            f'{self.frame.columns} nor is the position column specified. '
                            "Since the dataframe has a 'pixel' column, consider running "
                            f'pix2deg() before {transform_method.__name__}(). If you want '
                            'to calculate pixel transformations, you can do so by using '
                            f"{transform_method.__name__}(position_column='pixel'). "
                            f'Available dataframe columns are {self.frame.columns}',
                        )
                    raise pl.exceptions.ColumnNotFoundError(
                        "Neither 'position' is in the columns of the dataframe: "
                        f'{self.frame.columns} nor is the position column specified. '
                        f'Available dataframe columns are {self.frame.columns}',
                    )
            if transform_method.__name__ in {'pix2deg'}:
                if 'pixel' not in self.frame.columns and 'pixel_column' not in kwargs:
                    raise pl.exceptions.ColumnNotFoundError(
                        "Neither 'position' is in the columns of the dataframe: "
                        f'{self.frame.columns} nor is the pixel column specified. '
                        'You can specify the pixel column via: '
                        f'{transform_method.__name__}(pixel_column="name_of_your_pixel_column"). '
                        f'Available dataframe columns are {self.frame.columns}',
                    )

            if self.trial_columns is None:
                self.frame = self.frame.with_columns(transform_method(**kwargs))
            else:
                self.frame = pl.concat(
                    [
                        df.with_columns(transform_method(**kwargs))
                        for group, df in self.frame.groupby(self.trial_columns, maintain_order=True)
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
        window_length:
            The window size to use.
        degree:
            The degree of the polynomial to use.
        padding:
            The padding method to use. See ``savitzky_golay`` for details.

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
        method : str
            Computation method. See :func:`~transforms.pos2vel()` for details, default: fivepoint.
        **kwargs
            Additional keyword arguments to be passed to the :func:`~transforms.pos2vel()` method.

        Raises
        ------
        AttributeError
            If `gaze` is None or there are no gaze dataframes present in the `gaze` attribute, or
            if experiment is None.
        """
        self.transform('pos2vel', method=method, **kwargs)

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
        input_columns:
            Names of input columns to be merged into a single tuple column.
        output_column:
            Name of the resulting tuple column.
        """
        _check_component_columns(frame=self.frame, **{output_column: input_columns})

        self.frame = self.frame.with_columns(
            pl.concat_list([pl.col(component) for component in input_columns])
            .alias(output_column),
        ).drop(input_columns)

    def unnest(
            self,
            column: str,
            output_suffixes: list[str] | None = None,
            *,
            output_columns: list[str] | None = None,
    ) -> None:
        """Explode a column of type ``pl.List`` into one column for each list component.

        The input column will be dropped.

        Parameters
        ----------
        column:
            Name of input columns to be unnested into several component columns.
        output_columns:
            Name of the resulting tuple columns.
        output_suffixes:
            Suffixes to append to the column names.

        Raises
        ------
        ValueError
            If both output_columns and output_suffixes are specified.
            If number of output columns / suffixes does not match number of components.
            If output columns / suffixes are not unique.
        AttributeError
            If number of components is not 2, 4 or 6.
        """
        checks.check_is_mutual_exclusive(
            output_columns=output_columns,
            output_suffixes=output_suffixes,
        )
        _check_n_components(self.n_components)

        col_names = output_columns if output_columns is not None else []

        if output_columns is None and output_suffixes is None:
            if self.n_components == 2:
                output_suffixes = ['_x', '_y']
            elif self.n_components == 4:
                output_suffixes = ['_xl', '_yl', '_xr', '_yr']
            else:  # This must be 6 as we already have checked our n_components.
                output_suffixes = ['_xl', '_yl', '_xr', '_yr', '_xa', '_ya']

        if output_suffixes:
            col_names = [f'{column}{suffix}' for suffix in output_suffixes]

        if len(col_names) != self.n_components:
            raise ValueError(
                f'Number of output columns / suffixes ({len(col_names)}) '
                f'must match number of components ({self.n_components})',
            )
        if len(set(col_names)) != len(col_names):
            raise ValueError('Output columns / suffixes must be unique')

        self.frame = self.frame.with_columns(
            [
                pl.col(column).list.get(component_id).alias(col_names)
                for component_id, col_names in enumerate(col_names)
            ],
        ).drop(column)

    def copy(self) -> GazeDataFrame:
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
        """Check if experiment attribute has been set."""
        if self.experiment is None:
            raise AttributeError('experiment must not be None for this method to work')


def _check_component_columns(
        frame: pl.DataFrame,
        **kwargs: list[str],
) -> None:
    """Check if component columns are in valid format."""
    for component_type, columns in kwargs.items():
        if not isinstance(columns, list):
            raise TypeError(
                f'{component_type} must be of type list, but is of type {type(columns).__name__}',
            )

        for column in columns:
            if not isinstance(column, str):
                raise TypeError(
                    f'all elements in {component_type} must be of type str, '
                    f'but one of the elements is of type {type(column).__name__}',
                )

        if len(columns) not in [2, 4, 6]:
            raise ValueError(
                f'{component_type} must contain either 2, 4 or 6 columns, but has {len(columns)}',
            )

        for column in columns:
            if column not in frame.columns:
                raise pl.exceptions.ColumnNotFoundError(
                    f'column {column} from {component_type} is not available in dataframe',
                )

        if len(set(frame[columns].dtypes)) != 1:
            types_list = sorted([str(t) for t in set(frame[columns].dtypes)])
            raise ValueError(
                f'all columns in {component_type} must be of same type, but types are {types_list}',
            )


def _check_n_components(n_components: Any) -> None:
    """Check that n_components is either 2, 4 or 6."""
    if n_components not in {2, 4, 6}:
        raise AttributeError(f'n_components must be either 2, 4 or 6 but is {n_components}')


def _infer_n_components(frame: pl.DataFrame, column_specifiers: list[list[str]]) -> int | None:
    """Infer number of components from DataFrame.

    Method checks nested columns `pixel`, `position`, `velocity` and `acceleration` for number of
    components by getting their list lenghts, which must be equal for all else a ValueError is
    raised. Additionally, a list of list of column specifiers is checked for consistency.

    Parameters
    ----------
    frame: pl.DataFrame
        DataFrame to check.
    column_specifiers:
        List of list of column specifiers.

    Returns
    -------
    int or None
        Number of components

    Raises
    ------
    ValueError
        If number of components is not equal for all considered columns and rows.
    """
    all_considered_columns = ['pixel', 'position', 'velocity', 'acceleration']
    considered_columns = [column for column in all_considered_columns if column in frame.columns]

    list_lengths = {
        list_length
        for column in considered_columns
        for list_length in frame.get_column(column).list.lengths().unique().to_list()
    }

    for column_specifier_list in column_specifiers:
        list_lengths.add(len(column_specifier_list))

    if len(list_lengths) > 1:
        raise ValueError(f'inconsistent number of components inferred: {list_lengths}')

    if len(list_lengths) == 0:
        return None

    return next(iter(list_lengths))
