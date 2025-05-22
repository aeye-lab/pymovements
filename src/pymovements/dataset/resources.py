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
from typing import Literal

from pymovements._utils._dataclasses import asdict_factory


ContentType = Literal[
    'gaze',
    'precomputed_events',
    'precomputed_reading_measures',
]


@dataclass
class Resource:
    content: ContentType
    filename: str
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
        _asdict_factory = asdict_factory(exclude_none=exclude_none)
        data = asdict(self, dict_factory=_asdict_factory)
        return data


class Resources(tuple):
    def __new__(cls, *resources):
        if resources is None:
            return cls.__new__(tuple())
        return super(Resources, cls).__new__(cls, resources)

    def filter(self, content: ContentType | None = None) -> tuple[dict[str, str | None], ...]:
        if content is None:
            return self

        resources = [resource for resource in self if resource.content == content]
        return Resources(*resources)

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
            for content_dictionary in content_dictionaries:
                content_dictionary = deepcopy(content_dictionary)
                content_dictionary['content'] = content_type
                resource = Resource.from_dict(content_dictionary)
                resources.append(resource)

        return Resources(*resources)

    @staticmethod
    def from_dicts(
            dictionaries: list[dict[str, str | None]] | tuple[dict[str, str | None]] | None,
    ) -> Resources:
        if dictionaries is None:
            return Resources(None)

        resources = [
            Resource.from_dict(dictionary) for dictionary in dictionaries
        ]

        return Resources(*resources)

    def to_dicts(self) -> tuple[dict[str, str | None], ...] | None:
        return tuple([resource.to_dict() for resource in self])
