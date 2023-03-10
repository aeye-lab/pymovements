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
"""Module for event processing."""
from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

import polars as pl

from pymovements.events.event_properties import EVENT_PROPERTIES
from pymovements.events.events import EventDataFrame
from pymovements.exceptions import InvalidProperty
from pymovements.gaze.gaze_dataframe import GazeDataFrame


class EventProcessor:
    """Processes event and gaze dataframes.

    Attributes
    ----------
    event_properties: list[str]
        A list of property names.
    """

    def __init__(self, event_properties: str | list[str]):
        """Initialize processor with event property definitions.

        Parameters
        ----------
        event_properties:
            List of event property names.
        """
        if isinstance(event_properties, str):
            event_properties = [event_properties]

        for property_name in event_properties:
            if property_name not in EVENT_PROPERTIES:
                valid_properties = list(EVENT_PROPERTIES.keys())
                raise InvalidProperty(
                    property_name=property_name, valid_properties=valid_properties,
                )

        self.event_properties = event_properties

    def process(self, events: EventDataFrame) -> pl.DataFrame:
        """Process event dataframe.

        Parameters
        ----------
        events:
            Event data to process event properties from.

        Returns
        -------
        pl.DataFrame
            :py:class:`polars.DataFrame` with properties as columns and rows refering to the rows in
            the source dataframe.

        Raises
        ------
        InvalidProperty
            If ``property_name`` is not a valid property. See
            :py:mod:`pymovements.events.event_properties` for an overview of supported properties.
        """
        property_expressions: dict[str, Callable[[], pl.Expr]] = {
            property_name: EVENT_PROPERTIES[property_name]
            for property_name in self.event_properties
        }

        expression_list = [
            property_expression().alias(property_name)
            for property_name, property_expression in property_expressions.items()
        ]
        result = events.frame.select(expression_list)
        return result


class EventGazeProcessor:
    """Processes event and gaze dataframes.

    Attributes
    ----------
    event_properties: list[str]
        A list of property names.
    """

    def __init__(self, event_properties: str | list[str]):
        """Initialize processor with event property definitions.

        Parameters
        ----------
        event_properties:
            List of event property names.
        """
        if isinstance(event_properties, str):
            event_properties = [event_properties]

        for property_name in event_properties:
            if property_name not in EVENT_PROPERTIES:
                valid_properties = list(EVENT_PROPERTIES.keys())
                raise InvalidProperty(
                    property_name=property_name, valid_properties=valid_properties,
                )

        self.event_properties = event_properties

    def process(
            self,
            events: EventDataFrame,
            gaze: GazeDataFrame,
            identifiers: str | list[str],
    ) -> pl.DataFrame:
        """Process event and gaze dataframe.

        Parameters
        ----------
        events:
            Event data to process event properties from.
        gaze:
            Gaze data to process event properties from.
        identifiers:
            Column names to join on events and gaze dataframes.

        Returns
        -------
        pl.DataFrame
            :py:class:`polars.DataFrame` with properties as columns and rows refering to the rows in
            the source dataframe.

        Raises
        ------
        ValueError
            If list of identifiers is empty.
        InvalidProperty
            If ``property_name`` is not a valid property. See
            :py:mod:`pymovements.events.event_properties` for an overview of supported properties.
        """
        if isinstance(identifiers, str):
            identifiers = [identifiers]
        if len(identifiers) == 0:
            raise ValueError('list of identifiers must not be empty')

        property_expressions: dict[str, Callable[..., pl.Expr]] = {
            property_name: EVENT_PROPERTIES[property_name]
            for property_name in self.event_properties
        }

        property_kwargs: dict[str, dict[str, Any]] = {
            property_name: {} for property_name in property_expressions.keys()
        }
        for property_name, property_expression in property_expressions.items():
            property_args = inspect.getfullargspec(property_expression).args
            if 'velocity_columns' in property_args:
                velocity_columns = tuple(gaze.velocity_columns[:2])
                property_kwargs[property_name]['velocity_columns'] = velocity_columns

            if 'position_columns' in property_args:
                position_columns = tuple(gaze.position_columns[:2])
                property_kwargs[property_name]['position_columns'] = position_columns

        result = (
            gaze.frame.join(events.frame, on=identifiers)
            .filter(pl.col('time').is_between(pl.col('onset'), pl.col('offset')))
            .groupby([*identifiers, 'name', 'onset', 'offset'])
            .agg(
                [
                    property_expression(**property_kwargs[property_name])
                    .alias(property_name)
                    for property_name, property_expression in property_expressions.items()
                ],
            )
        )
        return result
