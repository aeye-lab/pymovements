# Copyright (c) 2022-2024 The pymovements Project Authors
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
"""Provides the EventDataFrame class."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import polars as pl

from pymovements.events.properties import duration
from pymovements.utils import checks


class EventDataFrame:
    """A DataFrame for event data.

    Each row has at least an event name with its onset and offset specified.

    Parameters
    ----------
    data: pl.DataFrame | None
        A dataframe to be transformed to a polars dataframe. This argument is mutually
        exclusive with all the other arguments. (default: None)
    name: str | list[str] | None
        Name of events. (default: None)
    onsets: list[int] | np.ndarray | None
        List of onsets. (default: None)
    offsets: list[int] | np.ndarray | None
        List of offsets. (default: None)
    trials: list[int | float | str] | np.ndarray | None
        List of trial identifiers. (default: None)
    trial_columns: list[str] | str | None
        List of trial columns in passed dataframe.

    Raises
    ------
    ValueError
        If list of onsets is passed but not a list of offsets, or vice versa, or if length of
        onsets does not match length of offsets.
    """

    _minimal_schema = {'name': pl.Utf8, 'onset': pl.Int64, 'offset': pl.Int64}

    def __init__(
            self,
            data: pl.DataFrame | None = None,
            name: str | list[str] | None = None,
            onsets: list[int] | np.ndarray | None = None,
            offsets: list[int] | np.ndarray | None = None,
            trials: list[int | float | str] | np.ndarray | None = None,
            trial_columns: list[str] | str | None = None,
    ):
        self.trial_columns: list[str] | None  # otherwise mypy gets confused.

        if data is not None:
            checks.check_is_mutual_exclusive(data=data, onsets=onsets)
            checks.check_is_mutual_exclusive(data=data, offsets=offsets)
            checks.check_is_mutual_exclusive(data=data, name=name)
            checks.check_is_mutual_exclusive(data=data, name=trials)

            data = data.clone()
            data = self._add_minimal_schema_columns(data)
            data_dict = data.to_dict()

            if isinstance(trial_columns, str):
                self.trial_columns = [trial_columns]
            else:
                self.trial_columns = trial_columns

            self._additional_columns = [
                column_name for column_name in data_dict.keys()
                if column_name not in self._minimal_schema
            ]

        else:
            # Make sure that if either onsets or offsets is None, the other one is None too.
            checks.check_is_none_is_mutual(onsets=onsets, offsets=offsets)

            # Make sure lengths of onsets and offsets are equal.
            if onsets is not None:

                # mypy does not get that offsets cannot be None (l. 87)
                assert offsets is not None

                checks.check_is_length_matching(onsets=onsets, offsets=offsets)
                # In case name is given as a list, check that too.
                if isinstance(name, Sequence) and not isinstance(name, str):
                    checks.check_is_length_matching(onsets=onsets, name=name)

                # These reassignments are necessary for a correct conversion into a dataframe.
                if len(onsets) == 0:
                    name = []
                if name is None:
                    name = ''
                if isinstance(name, str):
                    name = [name] * len(onsets)

                data_dict = {
                    'name': pl.Series(name, dtype=pl.Utf8),
                    'onset': pl.Series(onsets, dtype=pl.Int64),
                    'offset': pl.Series(offsets, dtype=pl.Int64),
                }

                if trials is not None:
                    data_dict['trial'] = pl.Series('trial', trials)
                    self.trial_columns = ['trial']
                else:
                    self.trial_columns = None

            else:
                data_dict = {
                    'name': pl.Series([], dtype=pl.Utf8),
                    'onset': pl.Series([], dtype=pl.Int64),
                    'offset': pl.Series([], dtype=pl.Int64),
                }
                self.trial_columns = None

        self.frame = pl.DataFrame(data=data_dict, schema_overrides=self._minimal_schema)

        # Ensure column order: trial columns, name, onset, offset.
        if self.trial_columns is not None:
            self.frame = self.frame.select([*self.trial_columns, *self._minimal_schema.keys()])

        if 'duration' not in self.frame.columns:
            self._add_duration_property()

    @property
    def schema(self) -> pl.type_aliases.SchemaDict:
        """Schema of event dataframe."""
        return self.frame.schema

    def __len__(self) -> int:
        """Get number of events in dataframe."""
        return self.frame.__len__()

    def __getitem__(self, *args: Any, **kwargs: Any) -> Any:
        """Get item."""
        return self.frame.__getitem__(*args, **kwargs)

    @property
    def columns(self) -> list[str]:
        """List of column names."""
        return self.frame.columns

    def _add_duration_property(self) -> None:
        """Add duration property column to dataframe."""
        self.frame = self.frame.select([pl.all(), duration().alias('duration')])

    def add_event_properties(
            self,
            event_properties: pl.DataFrame,
            join_on: str | list[str],
    ) -> None:
        """Add new event properties into dataframe.

        Parameters
        ----------
        event_properties: pl.DataFrame
            Dataframe with new event properties.
        join_on: str | list[str]
            Columns to join event properties on.
        """
        self.frame = self.frame.join(event_properties, on=join_on, how='left')

    def add_trial_column(
            self,
            column: str | list[str],
            data: int | float | str | list[int | float | str] | None,
    ) -> None:
        """Add new trial columns with constant values.

        Parameters
        ----------
        column: str | list[str]
            The name(s) of the new trial column(s).
        data: int | float | str | list[int | float | str] | None
            The values to be used for filling the trial column(s). In case multiple columns are
            provided, data must be a list of values matching the provided column order.
        """
        # Create trial column dictionary to iterate over in select().
        if isinstance(column, str):
            trial_columns = {column: data}
        # In case a list of a single column is passed as an explicit value.
        elif len(column) == 1 and (isinstance(data, (int, float, str) or data is None)):
            trial_columns = {column[0]: data}
        else:
            if not isinstance(data, Sequence):
                raise TypeError(
                    'data must be passed as a list of values in case of providing multiple columns',
                )
            checks.check_is_length_matching(column=column, data=data)

            trial_columns = dict(zip(column, data))

        self.frame = self.frame.select(
            [
                pl.lit(column_data).alias(column_name) if not isinstance(column_data, int)
                # Enforce Int64 columns for integers.
                else pl.lit(column_data).alias(column_name).cast(pl.Int64)
                for column_name, column_data in trial_columns.items()
            ] + [pl.all()],
        )

    @property
    def event_property_columns(self) -> list[str]:
        """Event property columns for this dataframe.

        Returns
        -------
        list[str]
            List of event property columns.
        """
        event_property_columns = set(self.frame.columns)
        event_property_columns -= set(list(self._minimal_schema.keys()))
        event_property_columns -= set(self._additional_columns)
        return list(event_property_columns)

    def copy(self) -> EventDataFrame:
        """Return a copy of the EventDataFrame.

        Returns
        -------
        EventDataFrame
            A copy of the EventDataFrame.
        """
        return EventDataFrame(data=self.frame.clone())

    def _add_minimal_schema_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add minimal schema columns to :py:class:`polars.DataFrame` if they are missing.

        Parameters
        ----------
        df: pl.DataFrame
            A dataframe to be transformed to a polars dataframe.

        Returns
        -------
        pl.DataFrame
            A dataframe with minimal schema columns added.
        """
        if len(df) == 0:
            return pl.DataFrame(schema={**self._minimal_schema, **df.schema})

        df = df.select(
            [
                pl.lit(None).cast(column_type).alias(column_name)
                for column_name, column_type in self._minimal_schema.items()
                if column_name not in df.columns
            ] + [pl.all()],
        )
        return df
