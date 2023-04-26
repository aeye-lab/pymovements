# Copyright (c) 2022-2023 The pymovements Project Authors
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
"""Module for py:func:`pymovements.gaze.transforms.norm`"""
from __future__ import annotations

import polars as pl

from pymovements.gaze.transforms_pl.transforms_library import register_transform


@register_transform
def norm(
        *,
        columns: tuple[str, str],
) -> pl.Expr:
    r"""Take the norm of a 2D series.

    The norm is defined by :math:`\sqrt{x^2 + y^2}` with :math:`x` being the yaw component and
    :math:`y` being the pitch component of a coordinate.

    Parameters
    ----------
    columns:
        Columns to take norm of.
    """
    x = pl.col(columns[0])
    y = pl.col(columns[1])
    return (x.pow(2) + y.pow(2)).sqrt()
