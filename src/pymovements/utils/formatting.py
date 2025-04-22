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

from html import escape
from typing import Any
from uuid import uuid4

import polars as pl


class HTML:
    STYLE = """
    <style>
        .pm-section-list {
            margin: 0;
            padding: 0;
            font-family: sans-serif;
        }
        .pm-section {
            list-style: none;
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
            padding-left: 1em;
        }
        .pm-section-toggle:checked ~ .pm-section-details {
            display: block;
        }
    </style>
    """

    @staticmethod
    def repr(obj: Any, attrs: list[str] | None = None) -> str:
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

        title = type(obj).__name__
        sections = []
        for attr in attrs:
            attr_obj = getattr(obj, attr)
            sections.append(HTML.section(attr, attr_obj))

        html = HTML.STYLE
        html += HTML.section_list(title, sections)
        return html

    @staticmethod
    def section_list(title: str, sections: list[str]) -> str:
        html = escape(title)
        html += """<ul class="pm-section-list">"""
        for section in sections:
            html += section
        html += '</ul>'
        return html

    @staticmethod
    def section(title: str, obj: Any) -> str:
        section_id = uuid4()
        title = escape(title)

        if isinstance(obj, pl.DataFrame):
            inline_details = escape(f"DataFrame ({len(obj.columns)} columns, {len(obj)} rows)")
        else:
            inline_details = escape(repr(obj).replace('\n', ' ')[:50])

        if hasattr(obj, '_repr_html_'):
            details = obj._repr_html_()
        else:
            details = escape(repr(obj))

        return f"""
        <li class="pm-section">
            <input id="pm-{section_id}" class="pm-section-toggle" type="checkbox">
            <label for="pm-{section_id}" class="pm-section-label">{title}:</label>
            <div class="pm-section-inline-details">{inline_details}</div>
            <div class="pm-section-details">{details}</div>
        </li>
        """
