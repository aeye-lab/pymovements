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

from typing import Any


def asdict_factory(
    *,
    exclude_private: bool = True,
    exclude_none: bool = True,
):
    """Return asdict_factory for being used in dataclasses.asdict().

    Parameters
    ----------
    exclude_private: bool
        Exclude attributes that start with ``_``.
    exclude_none: bool
        Exclude attributes that are either ``None`` or that are objects that evaluate to
        ``False`` (e.g., ``[]``, ``{}``, ``EyeTracker()``). Attributes of type ``bool``,
        ``int``, and ``float`` are not excluded.

    Returns
    -------
    dict[str, Any]
        Dictionary representation of dataset definition.
    """
    def _is_included(key: str, value: Any):
        # Exclude private fields from dictionary.
        if exclude_private and key.startswith('_'):
            return False
        # Exclude fields that evaluate to False (False, None, [], {}).
        if exclude_none and not isinstance(value, (bool, int, float)) and not value:
            return False
        # Otherwise include item.
        return True

    def _dict_factory(data: list[tuple[str, Any]]) -> dict[str, Any]:
        return dict(item for item in data if _is_included(*item))

    return _dict_factory
