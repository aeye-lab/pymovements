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
from typing import Any

import polars as pl

from pymovements.gaze import transforms
from pymovements.gaze.experiment import Experiment


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
        time_column:
            The name if the timestamp column in the input data frame.
        pixel_columns:
            The name of the pixel position columns in the input data frame.
        position_columns:
            The name of the dva position columns in the input data frame.
        velocity_columns:
            The name of the dva velocity columns in the input data frame.
        acceleration_columns:
            The name of the dva acceleration columns in the input data frame.

        Notes
        -----
        About using the arguments ``pixel_columns``, ``position_columns``, ``velocity_columns``,
        and ``acceleration_columns``:

        By passing a list of columns as any of these arguments, these columns will be merged into a
        single column with the corresponding name , e.g. using `pixel_columns` will merge the
        respective columns into the column `pixel`.

        The supported number of component columns with the expected order are:

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

        n_components = None
        if pixel_columns is not None:
            _check_component_columns(
                frame=self.frame,
                pixel_columns=pixel_columns,
            )
            self.nest(
                input_columns=pixel_columns,
                output_column='pixel',
            )
            n_components = len(pixel_columns)

        if position_columns is not None:
            _check_component_columns(
                frame=self.frame,
                position_columns=position_columns,
            )

            self.nest(
                input_columns=position_columns,
                output_column='position',
            )
            n_components = len(position_columns)

        if velocity_columns is not None:
            _check_component_columns(
                frame=self.frame,
                velocity_columns=velocity_columns,
            )

            self.nest(
                input_columns=velocity_columns,
                output_column='velocity',
            )
            n_components = len(velocity_columns)

        if acceleration_columns is not None:
            _check_component_columns(
                frame=self.frame,
                acceleration_columns=acceleration_columns,
            )

            self.nest(
                input_columns=acceleration_columns,
                output_column='acceleration',
            )
            n_components = len(acceleration_columns)

        self.n_components = n_components
        self.experiment = experiment

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
        self.frame = self.frame.with_columns(
            pl.concat_list([pl.col(component) for component in input_columns])
            .alias(output_column),
        ).drop(input_columns)

    def unnest(
            self,
            column: str,
            output_columns: list[str],
    ) -> None:
        """Explode a column of type ``pl.List`` into one column for each list component.

        The input column will be dropped.

        Parameters
        ----------
        column:
            Name of input columns to be unnested into several component columns.
        output_columns:
            Name of the resulting tuple columns.
        """
        self.frame = self.frame.with_columns(
            [
                pl.col(column).list.get(component_id).alias(output_column)
                for component_id, output_column in enumerate(output_columns)
            ],
        ).drop(column)

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
