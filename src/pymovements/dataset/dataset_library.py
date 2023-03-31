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
"""DatasetLibrary module."""
from __future__ import annotations

from typing import Type
from typing import TypeVar

from pymovements.dataset.dataset_definition import DatasetDefinition


class DatasetLibrary:
    """Provides access by name to :py:class:`~pymovements.DatasetDefinition`.

    Attributes
    ----------
    definitions:
        Dictionary of :py:class:`~pymovements.DatasetDefinition`.
    """

    definitions: dict[str, type[DatasetDefinition]] = {}

    @classmethod
    def add(cls, definition: type[DatasetDefinition]) -> None:
        """Add :py:class:`~pymovements.DatasetDefinition` to library.

        Parameter
        ---------
        definition
            The :py:class:`~pymovements.DatasetDefinition` to add to the library.
        """
        cls.definitions[definition.name] = definition

    @classmethod
    def get(cls, name: str) -> type[DatasetDefinition]:
        """Get :py:class:`~pymovements.DatasetDefinition` py name.

        Parameter
        ---------
        name
            Name of the :py:class:`~pymovements.DatasetDefinition` in the library.
        """
        return cls.definitions[name]


DatsetDefinitionClass = TypeVar('DatsetDefinitionClass', bound=Type[DatasetDefinition])


def register_dataset(cls: DatsetDefinitionClass) -> DatsetDefinitionClass:
    """Register a public dataset definition."""
    DatasetLibrary.add(cls)
    return cls
