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

import pymovements as pm  # pylint: disable=cyclic-import
from pymovements.events.frame import EventDataFrame
from pymovements.events.properties import EVENT_PROPERTIES
from pymovements.exceptions import InvalidProperty


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

    def __init__(
            self,
            event_properties: str | tuple[str, dict[str, Any]]
            | list[str | tuple[str, dict[str, Any]]],
    ):
        """Initialize processor with event property definitions.

        Parameters
        ----------
        event_properties:
            List of event property names.
        """
        if isinstance(event_properties, (str, tuple)):
            event_properties = [event_properties]

        event_properties_with_kwargs = []
        for event_property in event_properties:
            if isinstance(event_property, str):
                property_name = event_property
                property_kwargs = {}
            else:
                property_name = event_property[0]
                property_kwargs = event_property[1]

            if property_name not in EVENT_PROPERTIES:
                valid_properties = list(EVENT_PROPERTIES.keys())
                raise InvalidProperty(
                    property_name=property_name, valid_properties=valid_properties,
                )

            event_properties_with_kwargs.append((property_name, property_kwargs))

        self.event_properties: list[tuple[str, dict[str, Any]]] = event_properties_with_kwargs

    def process(
            self,
            events: EventDataFrame,
            gaze: pm.GazeDataFrame,
            identifiers: str | list[str],
            name: str | None = None,
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
        name:
            Process only events that match the name.

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
        RuntimeError
            If specified event name ``name`` is missing from ``events``.
        """
        if isinstance(identifiers, str):
            trial_identifiers = [identifiers]
        else:
            trial_identifiers = identifiers

        if len(trial_identifiers) == 0:
            raise ValueError('list of identifiers must not be empty')

        property_expressions: list[Callable[..., pl.Expr]] = [
            EVENT_PROPERTIES[property_name] for property_name, _ in self.event_properties
        ]

        property_names: list[str] = [property_name for property_name, _ in self.event_properties]

        property_kwargs: list[dict[str, Any]] = [
            property_kwargs for _, property_kwargs in self.event_properties
        ]

        for property_id, property_expression in enumerate(property_expressions):
            property_args = inspect.getfullargspec(property_expression).kwonlyargs

            if 'position_column' in property_args:
                property_kwargs[property_id]['position_column'] = 'position'

            if 'velocity_column' in property_args:
                property_kwargs[property_id]['velocity_column'] = 'velocity'

        # Each event is uniquely defined by a list of trial identifiers,
        # a name and its on- and offset.
        event_identifiers = [*trial_identifiers, 'name', 'onset', 'offset']

        joined_frame = gaze.frame.join(events.frame, on=trial_identifiers)
        if name is not None:
            joined_frame = joined_frame.filter(pl.col('name').str.contains(f'^{name}$'))

        if len(joined_frame) == 0:
            raise RuntimeError(f'No events with name "{name}" found in data frame')

        result = (
            joined_frame
            .filter(pl.col('time').is_between(pl.col('onset'), pl.col('offset')))
            .groupby(event_identifiers, maintain_order=True)
            .agg(
                [
                    this_property_expression(**this_property_kwargs)
                    .alias(this_property_name)
                    for this_property_name, this_property_expression, this_property_kwargs,
                    in zip(property_names, property_expressions, property_kwargs)
                ],
            )
        )
        return result
