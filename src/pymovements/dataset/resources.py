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
class Resources:
    resources: tuple[dict[str, str]] | None = None

    def get(self, content: ContentType | None = None) -> tuple[dict[str, str]]:
        if not self.resources:
            return tuple()

        if content is None:
            return self.resources

        return tuple(
            [
                resource for resource in self.resources if resource['content'] == content
            ],
        )

    @staticmethod
    def from_dict(dictionary: dict[str, list[dict[str, str]] | tuple[dict[str, str]]] | None):
        if dictionary is None:
            return Resources(None)

        resources = []
        for content, content_resources in dictionary.items():
            for content_resource in content_resources:
                print(content_resource)
                resource = deepcopy(content_resource)
                resource['content'] = content
                resources.append(resource)

        return Resources(tuple(resources))

    @staticmethod
    def from_dicts(dictionaries: list[dict[str, str]] | tuple[dict[str, str]] | None):
        if dictionaries is None:
            return Resources(None)

        return Resources(tuple(dictionaries))

    def to_tuple_of_dicts(self) -> tuple[dict[str, str]] | None:
        return self.resources
