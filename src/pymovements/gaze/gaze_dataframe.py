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

import polars as pl

from pymovements.gaze.experiment import Experiment
from pymovements.gaze.transforms import pos2acc


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

        if time_column is not None:
            self.frame = self.frame.rename({time_column: 'time'})

        n_components = None
        if pixel_columns is not None:
            _check_component_columns(
                frame=self.frame,
                pixel_columns=pixel_columns,
            )
            self.merge_component_columns_into_tuple_column(
                input_columns=pixel_columns,
                output_column='pixel',
            )
            n_components = len(pixel_columns)

        if position_columns is not None:
            _check_component_columns(
                frame=self.frame,
                position_columns=position_columns,
            )

            self.merge_component_columns_into_tuple_column(
                input_columns=position_columns,
                output_column='position',
            )
            n_components = len(position_columns)

        if velocity_columns is not None:
            _check_component_columns(
                frame=self.frame,
                velocity_columns=velocity_columns,
            )

            self.merge_component_columns_into_tuple_column(
                input_columns=velocity_columns,
                output_column='velocity',
            )
            n_components = len(velocity_columns)

        if acceleration_columns is not None:
            _check_component_columns(
                frame=self.frame,
                acceleration_columns=acceleration_columns,
            )

            self.merge_component_columns_into_tuple_column(
                input_columns=acceleration_columns,
                output_column='acceleration',
            )
            n_components = len(acceleration_columns)

        self.n_components = n_components
        self.experiment = experiment

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
        self._check_experiment()
        # mypy does not get that experiment now cannot be None anymore
        assert self.experiment is not None

        # this is just a work-around until GazeDataFrame.transform() is implemented
        if 'pixel' in self.frame.columns:
            pixel_columns = [
                '__x_left_pix__', '__y_left_pix__',
                '__x_right_pix__', '__y_right_pix__',
                '__x_avg_pix__', '__y_avg_pix__',
            ][:self.n_components]
            self.explode('pixel', pixel_columns)
        else:
            raise pl.exceptions.ColumnNotFoundError(
                f'Column \'pixel\' not found. Available columns are: {self.frame.columns}',
            )

        pixel_positions = self.frame.select(pixel_columns)
        dva_positions = self.experiment.screen.pix2deg(pixel_positions.to_numpy())

        dva_columns = self._pixel_to_dva_position_columns(pixel_columns)
        self.frame = self.frame.with_columns(
            [
                pl.Series(name=dva_column_name, values=dva_positions[:, dva_column_id])
                for dva_column_id, dva_column_name in enumerate(dva_columns)
            ],
        )

        self.merge_component_columns_into_tuple_column(
            input_columns=dva_columns,
            output_column='position',
        )

        # this is just a work-around until merged columns are standard behavior
        self.merge_component_columns_into_tuple_column(
            input_columns=pixel_columns,
            output_column='pixel',
        )

    def pos2acc(
            self,
            window_length: int = 7,
            degree: int = 2,
            mode: str = 'interp',
            cval: float = 0.0,
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
        mode:
            The padding mode to use.
        cval:
            A constant value for padding.

        Raises
        ------
        AttributeError
            If `gaze` is None or there are no gaze dataframes present in the `gaze` attribute, or
            if experiment is None.
        """
        self._check_experiment()
        # mypy does not get that experiment now cannot be None anymore
        assert self.experiment is not None

        # this is just a work-around until merged columns are standard behavior
        if 'position' in self.frame.columns:
            position_columns = [
                '__x_left_pos__', '__y_left_pos__',
                '__x_right_pos__', '__y_right_pos__',
                '__x_avg_pos__', '__y_avg_pos__',
            ][:self.n_components]
            self.explode('position', position_columns)
        else:
            raise pl.exceptions.ColumnNotFoundError(
                f'Column \'position\' not found. Available columns are: {self.frame.columns}',
            )

        positions = self.frame.select(position_columns)
        acceleration = pos2acc(
            positions.to_numpy(),
            sampling_rate=self.experiment.sampling_rate,
            window_length=window_length,
            degree=degree,
            mode=mode,
            cval=cval,
        )

        acceleration_columns = self._position_to_acceleration_columns(position_columns)
        self.frame = self.frame.with_columns(
            [
                pl.Series(name=velocity_column_name, values=acceleration[:, column_id])
                for column_id, velocity_column_name in enumerate(acceleration_columns)
            ],
        )

        self.merge_component_columns_into_tuple_column(
            input_columns=acceleration_columns,
            output_column='acceleration',
        )

        # this is just a work-around until merged columns are standard behavior
        self.merge_component_columns_into_tuple_column(
            input_columns=position_columns,
            output_column='position',
        )

    def pos2vel(self, method: str = 'smooth', **kwargs: int | float | str) -> None:
        """Compute gaze velocity in dva/s from dva position coordinates.

        This method requires a properly initialized :py:attr:`~.GazeDataFrame.experiment` attribute.

        After success, the gaze dataframe is extended by the resulting velocity columns.

        Parameters
        ----------
        method : str
            Computation method. See :func:`~transforms.pos2vel()` for details, default: smooth.
        **kwargs
            Additional keyword arguments to be passed to the :func:`~transforms.pos2vel()` method.

        Raises
        ------
        AttributeError
            If `gaze` is None or there are no gaze dataframes present in the `gaze` attribute, or
            if experiment is None.
        """
        self._check_experiment()
        # mypy does not get that experiment now cannot be None anymore
        assert self.experiment is not None

        # this is just a work-around until merged columns are standard behavior
        if 'position' in self.frame.columns:
            position_columns = [
                '__x_left_pos__', '__y_left_pos__',
                '__x_right_pos__', '__y_right_pos__',
                '__x_avg_pos__', '__y_avg_pos__',
            ][:self.n_components]
            self.explode('position', position_columns)
        else:
            raise pl.exceptions.ColumnNotFoundError(
                f'Column \'position\' not found. Available columns are: {self.frame.columns}',
            )

        positions = self.frame.select(position_columns)
        velocities = self.experiment.pos2vel(positions.to_numpy(), method=method, **kwargs)

        velocity_columns = self._position_to_velocity_columns(position_columns)
        self.frame = self.frame.with_columns(
            [
                pl.Series(name=velocity_column_name, values=velocities[:, column_id])
                for column_id, velocity_column_name in enumerate(velocity_columns)
            ],
        )

        self.merge_component_columns_into_tuple_column(
            input_columns=velocity_columns,
            output_column='velocity',
        )

        # this is just a work-around until merged columns are standard behavior
        self.merge_component_columns_into_tuple_column(
            input_columns=position_columns,
            output_column='position',
        )

    @property
    def schema(self) -> pl.type_aliases.SchemaDict:
        """Schema of event dataframe."""
        return self.frame.schema

    @property
    def columns(self) -> list[str]:
        """List of column names."""
        return self.frame.columns

    @property
    def acceleration_columns(self) -> list[str]:
        """Acceleration columns (in degrees of visual angle per second^2) of dataframe."""
        acceleration_columns = list(set(self.valid_acceleration_columns) & set(self.frame.columns))
        return acceleration_columns

    @property
    def velocity_columns(self) -> list[str]:
        """Velocity columns (in degrees of visual angle per second) of dataframe."""
        velocity_columns = list(set(self.valid_velocity_columns) & set(self.frame.columns))
        return velocity_columns

    @property
    def pixel_position_columns(self) -> list[str]:
        """Pixel position columns for this dataset."""
        pixel_position_columns = set(self.valid_pixel_position_columns) & set(self.frame.columns)
        return list(pixel_position_columns)

    @property
    def position_columns(self) -> list[str]:
        """Position columns (in degrees of visual angle) for this dataset."""
        position_columns = set(self.valid_position_columns) & set(self.frame.columns)
        return list(position_columns)

    def merge_component_columns_into_tuple_column(
            self,
            input_columns: list[str],
            output_column: str,
    ) -> None:
        """Merge component columns into a single tuple columns.

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

    def explode(
            self,
            column: str,
            output_columns: list[str],
    ) -> None:
        """Explode a column of type ``pl.List`` into one column for each list component.

        The input column will be dropped.

        Parameters
        ----------
        column:
            Name of input columns to be exploded into several component columns.
        output_columns:
            Name of the resulting tuple columns.
        """
        self.frame = self.frame.with_columns(
            [
                pl.col(column).list.get(component_id).alias(output_column)
                for component_id, output_column in enumerate(output_columns)
            ],
        ).drop(column)

    @staticmethod
    def _pixel_to_dva_position_columns(columns: list[str]) -> list[str]:
        """Get corresponding dva position columns from pixel position columns."""
        return [
            column.replace('pixel', 'position').replace('pix', 'pos')
            for column in columns if 'pix' in column
        ]

    @staticmethod
    def _position_to_acceleration_columns(columns: list[str]) -> list[str]:
        """Get corresponding acceleration columns from dva position columns."""
        return [
            column.replace('position', 'acceleration').replace('pos', 'acc')
            for column in columns if 'pos' in column
        ]

    @staticmethod
    def _position_to_velocity_columns(columns: list[str]) -> list[str]:
        """Get corresponding velocity columns from dva position columns."""
        return [
            column.replace('position', 'velocity').replace('pos', 'vel')
            for column in columns if 'pos' in column
        ]

    def _check_experiment(self) -> None:
        """Check if experiment attribute has been set."""
        if self.experiment is None:
            raise AttributeError('experiment must be specified for this method to work')


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
