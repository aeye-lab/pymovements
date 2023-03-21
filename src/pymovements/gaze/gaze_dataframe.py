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


class GazeDataFrame:
    """A DataFrame for gaze time series data.

    Each row is a sample at a specific timestep.
    Each column is a channel in the gaze time series.
    """

    _valid_pixel_position_columns = [
        'x_pix', 'y_pix',
        'x_left_pix', 'y_left_pix',
        'x_right_pix', 'y_right_pix',
    ]

    _valid_position_columns = [
        'x_pos', 'y_pos',
        'x_left_pos', 'y_left_pos',
        'x_right_pos', 'y_right_pos',
    ]

    _valid_velocity_columns = [
        'x_vel', 'y_vel',
        'x_left_vel', 'y_left_vel',
        'x_right_vel', 'y_right_vel',
    ]

    def __init__(
            self,
            data: pl.DataFrame | None = None,
            experiment: Experiment | None = None,
    ):
        """Initialize a :py:class:`pymovements.gaze.gaze_dataframe.GazeDataFrame`.

        Parameters
        ----------
        data: pl.DataFrame
            A dataframe to be transformed to a polars dataframe.
        experiment : Experiment
            The experiment definition.
        """
        if data is None:
            data = pl.DataFrame()

        self.frame = data.clone()
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

        pix_position_columns = self.pixel_position_columns
        if not pix_position_columns:
            raise AttributeError(
                'No valid pixel position columns found.'
                f' Valid pixel position columns are: {self._valid_pixel_position_columns}.'
                f' Available columns are: {self.frame.columns}.',
            )

        dva_position_columns = self._pixel_to_dva_position_columns(pix_position_columns)

        pixel_positions = self.frame.select(pix_position_columns)
        dva_positions = self.experiment.screen.pix2deg(pixel_positions.to_numpy())

        self.frame = self.frame.with_columns(
            [
                pl.Series(name=dva_column_name, values=dva_positions[:, dva_column_id])
                for dva_column_id, dva_column_name in enumerate(dva_position_columns)
            ],
        )

    def pos2vel(self, method: str = 'smooth', **kwargs) -> None:
        """Compute gaze velocites in dva/s from dva position coordinates.

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

        position_columns = self.position_columns
        if not position_columns:
            raise AttributeError(
                'No valid position columns found.'
                f' Valid position columns are: {self._valid_position_columns}.'
                f' Available columns are: {self.frame.columns}.',
            )
        velocity_columns = self._position_to_velocity_columns(position_columns)

        positions = self.frame.select(position_columns)

        velocities = self.experiment.pos2vel(positions.to_numpy(), method=method, **kwargs)

        self.frame = self.frame.with_columns(
            [
                pl.Series(name=velocity_column_name, values=velocities[:, column_id])
                for column_id, velocity_column_name in enumerate(velocity_columns)
            ],
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
    def velocity_columns(self) -> list[str]:
        """Velocity columns (in degrees of visual angle per second) of dataframe."""
        velocity_columns = list(set(self._valid_velocity_columns) & set(self.frame.columns))
        return velocity_columns

    @property
    def pixel_position_columns(self) -> list[str]:
        """Pixel position columns for this dataset."""
        pixel_position_columns = set(self._valid_pixel_position_columns) & set(self.frame.columns)
        return list(pixel_position_columns)

    @property
    def position_columns(self) -> list[str]:
        """Position columns (in degrees of visual angle) for this dataset."""
        position_columns = set(self._valid_position_columns) & set(self.frame.columns)
        return list(position_columns)

    @staticmethod
    def _pixel_to_dva_position_columns(columns: list[str]) -> list[str]:
        """Get corresponding dva position columns from pixel position columns."""
        return [
            column.replace('_pix', '_pos')
            for column in columns
            if column.endswith('_pix')
        ]

    @staticmethod
    def _position_to_velocity_columns(columns: list[str]) -> list[str]:
        """Get corresponding velocity columns from dva position columns."""
        return [
            column.replace('_pos', '_vel')
            for column in columns
            if column.endswith('_pos')
        ]

    def _check_experiment(self) -> None:
        """Check if experiment attribute has been set."""
        if self.experiment is None:
            raise AttributeError('experiment must be specified for this method to work.')
