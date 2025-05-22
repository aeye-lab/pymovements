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
from dataclasses import dataclass
from typing import Literal


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


@dataclass
class Resources:
    resources: tuple[Resource, ...] | None = None

    def get(self, content: ContentType | None = None) -> tuple[dict[str, str | None], ...]:
        if not self.resources:
            return tuple()

        if content is None:
            return self.resources

        return tuple(
            [
                resource for resource in self.resources if resource.content == content
            ],
        )

    @staticmethod
    def from_dict(
            dictionary: dict[str, list[dict[str, str | None]]]
                        | dict[str, tuple[dict[str, str | None]], ...]
                        | None,
    ) -> Resources:
        if dictionary is None:
            return Resources(None)

        resources = []
        for content_type, content_dictionaries in dictionary.items():
            for content_dictionary in content_dictionaries:
                content_dictionary = deepcopy(content_dictionary)
                content_dictionary['content'] = content_type
                resource = Resource.from_dict(content_dictionary)
                resources.append(resource)

        return Resources(tuple(resources))

    @staticmethod
    def from_dicts(
            dictionaries: list[dict[str, str | None]] | tuple[dict[str, str | None]] | None,
    ) -> Resources:
        if dictionaries is None:
            return Resources(None)

        resources = [
            Resource.from_dict(dictionary) for dictionary in dictionaries
        ]

        return Resources(tuple(resources))

    def to_tuple_of_dicts(self) -> tuple[dict[str, str | None], ...] | None:
        return self.resources
