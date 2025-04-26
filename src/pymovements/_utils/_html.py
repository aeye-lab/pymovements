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
"""Provides functions for generating HTML representations of objects for Jupyter notebooks."""
from __future__ import annotations

from collections.abc import Callable
from html import escape
from typing import TypeVar
from uuid import uuid4

import polars as pl


STYLE = """
<style>
    .pm-section-list {
        margin: 0;
        padding: 0;
        font-family: sans-serif;
    }
    .pm-section {
        list-style: none;
        padding-top: 0.5em;
    }
    .pm-section-title {
        font-size: 120%;
        font-weight: bold;
    }
    .pm-section-toggle {
        display: none;
    }
    .pm-section-label {
        cursor: pointer;
        font-weight: bold;
    }
    .pm-section-label:before {
        display: inline-block;
        content: "►";
    }
    .pm-section-toggle:checked + .pm-section-label:before {
        content: "▼";
    }
    .pm-section-inline-details {
        display: inline-block;
    }
    .pm-section-details {
        display: none;
    }
    .pm-section-toggle:checked ~ .pm-section-details {
        display: block;
    }
</style>
"""


T = TypeVar('T')


def html_repr(attrs: list[str] | None = None) -> Callable[[T], T]:
    """Add an HTML representation to the class for Jupyter notebooks.

    Parameters
    ----------
    attrs : list[str] | None
        List of attributes to include in the HTML representation.
        If None, all public attributes are included.

    Returns
    -------
    Callable[[T], T]
        Decorator function that adds the HTML representation to the decorated class.
    """

    def decorator(cls: T) -> T:
        setattr(cls, '_repr_html_', lambda self: _obj_html(self, attrs))
        return cls

    return decorator


def _obj_html(obj: object, attrs: list[str] | None = None) -> str:
    if attrs is None:
        attrs = []
        for attr in dir(obj):
            # Skip private attributes
            if attr.startswith('_'):
                continue
            # Skip properties that raise errors
            try:
                value = getattr(obj, attr)
            except BaseException:
                continue
            # Skip methods
            if not callable(value):
                attrs.append(attr)

    title = escape(type(obj).__name__)

    sections = []
    for attr in attrs:
        attr_obj = getattr(obj, attr)
        sections.append(_attr_html(attr, attr_obj))

    return f"""
    {STYLE}
    <span class="pm-section-title">{title}</span>
    <ul class="pm-section-list">
    {"".join(sections)}
    </ul>
    """


def _attr_html(name: str, obj: object) -> str:
    section_id = uuid4()
    name = escape(name)

    inline_details = _attr_inline_details_html(obj)
    details = _attr_details_html(obj)

    return f"""
    <li class="pm-section">
        <input id="pm-{section_id}" class="pm-section-toggle" type="checkbox">
        <label for="pm-{section_id}" class="pm-section-label">{name}:</label>
        <div class="pm-section-inline-details">{inline_details}</div>
        <div class="pm-section-details">{details}</div>
    </li>
    """


def _attr_inline_details_html(obj: object) -> str:
    if isinstance(obj, pl.DataFrame):
        inline_details = f"DataFrame ({len(obj.columns)} columns, {len(obj)} rows)"
    elif isinstance(obj, list):
        inline_details = f"list ({len(obj)} items)"
    elif isinstance(obj, dict):
        inline_details = f"dict ({len(obj)} items)"
    elif len(repr(obj)) < 50:
        inline_details = repr(obj).replace('\n', ' ')
    else:
        inline_details = type(obj).__name__
    return escape(inline_details)


def _attr_details_html(obj: object) -> str:
    if isinstance(obj, list):
        details = '<ul>'
        num_shown = 2
        for item in obj[:num_shown]:
            # TODO: Limit recursion depth
            details += f'<li>{_attr_details_html(item)}</li>'
        if len(obj) > num_shown:
            details += f'<li>({len(obj) - 2} more)</li>'
        details += '</ul>'
    elif isinstance(obj, dict):
        details = '<ul>'
        num_shown = 2
        for key, value in list(obj.items())[:num_shown]:
            # TODO: Limit recursion depth
            details += f'<li><strong>{escape(str(key))}:</strong><br>'
            details += f'{_attr_details_html(value)}</li>'
        if len(obj) > num_shown:
            details += f'<li>({len(obj) - 2} more)</li>'
        details += '</ul>'
    elif hasattr(obj, '_repr_html_'):
        details = obj._repr_html_()
    else:
        details = escape(repr(obj))
    return details
