# Copyright (c) 2022-2023 The pymovements Project Authors
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
"""
This module holds all the main Event classes used for event detection.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import polars as pl
from typing_extensions import Protocol

from pymovements.base import DataFrame
from pymovements.utils import checks


class EventDataFrame(DataFrame):
    """A DataFrame for event data.

    Each row has at least an event name with its onset and offset specified.
    """

    _minimal_schema = {'name': pl.Utf8, 'onset': pl.Int64, 'offset': pl.Int64}

    def __init__(
            self,
            data: pl.DataFrame | None = None,
            name: str | list[str] | None = None,
            onsets: list[int] | None = None,
            offsets: list[int] | None = None,
    ):
        """Initialize an :py:class:`pymovements.events.event_dataframe.EventDataFrame`.

        This is a subclass of :py:class:`polars.DataFrame` with additional functionality tailored to
        gaze events.

        Parameters
        ----------
        dataframe: pl.DataFrame
            A dataframe to be transformed to a polars dataframe. This argument is mutually
            exclusive with all the other arguments.
        name: str
            Name of events
        onsets: list[int]
            List of onsets
        offsets; list[int]
            List of offsets

        Raises
        ------
        ValueError
            If list of onsets is passed but not a list of offsets, or vice versa, or if length of
            onsets does not match length of offsets.
        """
        if data is not None:
            checks.check_is_mutual_exclusive(data=data, onsets=onsets)
            checks.check_is_mutual_exclusive(data=data, offsets=offsets)
            checks.check_is_mutual_exclusive(data=data, name=name)

            data_dict = data.to_dict()

            # Add name column if it is missing.
            if 'name' not in data_dict:
                length = 0
                for dict_value in data_dict.values():
                    length = len(dict_value)
                    break
                data_dict['name'] = pl.Series([] * length, dtype=pl.Utf8)

            # Add onset column if it is missing.
            if 'onset' not in data_dict:
                length = 0
                for dict_value in data_dict.values():
                    length = len(dict_value)
                    break
                data_dict['onset'] = pl.Series([] * length, dtype=pl.Int64)

            # Add offset column if it is missing.
            if 'offset' not in data_dict:
                length = 0
                for dict_value in data_dict.values():
                    length = len(dict_value)
                    break
                data_dict['offset'] = pl.Series([] * length, dtype=pl.Int64)

        else:
            # Make sure that if either onsets or offsets is None, the other one is None too.
            checks.check_is_none_is_mutual(onsets=onsets, offsets=offsets)

            # Make sure lengths of onsets and offsets are equal.
            if onsets is not None:
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
            else:
                data_dict = {
                    'name': pl.Series([], dtype=pl.Utf8),
                    'onset': pl.Series([], dtype=pl.Int64),
                    'offset': pl.Series([], dtype=pl.Int64),
                }

        super().__init__(data=data_dict, schema_overrides=self._minimal_schema)


class EventDetectionCallable(Protocol):
    """Minimal interface to be implemented by all event detection methods."""

    def __call__(
            self,
            positions: list[list[float]] | list[tuple[float, float]] | np.ndarray,
            velocities: list[list[float]] | list[tuple[float, float]] | np.ndarray,
            **kwargs,
    ) -> EventDataFrame:
        """Minimal interface to be implemented by all event detection methods.

        Parameters
        ----------
        positions: array-like, shape (N, 2)
            Continuous 2D position time series
        velocities: array-like, shape (N, 2)
            Corresponding continuous 2D velocity time series.
        **kwargs:
            Additional keyword arguments for the specific event detection method.

        Returns
        -------
        EventDataFrame
            A dataframe with detected events as rows.
        """
