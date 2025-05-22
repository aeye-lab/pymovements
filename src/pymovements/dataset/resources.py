# Copyright (c) 2025 The pymovements Project Authors
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
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from dataclasses import dataclass
from typing import Literal, Any

ContentType = Literal[
    'gaze',
    'precomputed_events',
    'precomputed_reading_measures',
]


@dataclass
class Resource:
    content: ContentType
    filename: str | None = None
    url: str | None = None
    md5: str | None = None

    @staticmethod
    def from_dict(dictionary: dict[str, str]) -> Resource:
        if 'resource' in dictionary:
            url = dictionary['resource']
            dictionary = {key: value for key, value in dictionary.items() if key != 'resource'}
            dictionary['url'] = url

        return Resource(**dictionary)

    def to_dict(self, *, exclude_none: bool = True) -> dict[str, str | None]:
        data = asdict(self)

        # Delete fields that evaluate to False (False, None, [], {})
        if exclude_none:
            for key, value in list(data.items()):
                if not isinstance(value, (bool, int, float)) and not value:
                    del data[key]

        return data


class Resources(list):
    def __new__(cls, *resources):
        if resources is None:
            return cls.__new__(tuple())
        return super().__new__(cls, resources)

    def filter(self, content: ContentType | None = None) -> tuple[dict[str, str | None], ...]:
        if content is None:
            return self

        resources = [resource for resource in self if resource.content == content]
        return Resources(resources)

    @staticmethod
    def from_dict(
            dictionary: dict[str, list[dict[str, str | None]]]
        | dict[str, tuple[dict[str, str | None]], ...]
        | None,
    ) -> Resources:
        if dictionary is None:
            return Resources()

        resources = []
        for content_type, content_dictionaries in dictionary.items():
            if not content_dictionaries:
                continue
            for content_dictionary in content_dictionaries:
                content_dictionary = deepcopy(content_dictionary)
                content_dictionary['content'] = content_type
                resource = Resource.from_dict(content_dictionary)
                resources.append(resource)

        return Resources(resources)

    @staticmethod
    def from_dicts(
            dictionaries: list[dict[str, str | None]] | tuple[dict[str, str | None]] | None,
    ) -> Resources:
        if dictionaries is None:
            return Resources()

        resources = [Resource.from_dict(dictionary) for dictionary in dictionaries]

        return Resources(resources)

    def to_dicts(self, *, exclude_none: bool = True) -> list[dict[str, str | None]] | None:
        return [resource.to_dict(exclude_none=exclude_none) for resource in self]

    def has_content(self, content: str):
        # Check if any resources are actually set in dictionary.
        for resource in self:
            if resource.content == content:
                return True
        return False


class _HasResourcesIndexer:
    """Helper class for :py:meth:`~pymovements.dataset.DatasetDefinition.has_resources` property.

    Provides dynamic inference on the presence of any
    :py:meth:`~pymovements.dataset.DatasetDefinition.resources`.
    """

    def __init__(self) -> None:
        self._resources: Resources = {}

    def set_resources(self, resources: _Resources) -> None:
        """Set dataset definition resources for lookup."""
        self._resources = resources

    def __getitem__(self, key: str) -> bool:
        """Lookup if resources of specific content are set."""
        return self.__bool__() and self._resources.has_content(key)

    def __bool__(self) -> bool:
        """Lookup if resources of any content are set."""
        return bool(self._resources)

    def __eq__(self, other: Any) -> bool:
        """Return self == other.

        Automatically casts to bool if compared to a boolean.
        """
        if isinstance(other, bool):  # Needed to check equality against booleans.
            return self.__bool__() == other
        return super().__eq__(other)

    def __repr__(self) -> str:
        """Return string with boolean value wheter any resources are set."""
        return str(self.__bool__())
