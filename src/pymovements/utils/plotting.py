# Copyright (c) 2024 The pymovements Project Authors
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
"""Functions for plotting."""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Literal
from typing import Union

import matplotlib.pyplot
import numpy as np
from typing_extensions import TypeAlias

import PIL.Image

LinearSegmentedColormapType: TypeAlias = dict[
    Literal['red', 'green', 'blue', 'alpha'],
    Sequence[tuple[float, ...]],
]


DEFAULT_SEGMENTDATA: LinearSegmentedColormapType = {
    'red': [
        (0.0, 0.0, 0.0),
        (0.5, 1.0, 1.0),
        (1.0, 1.0, 1.0),
    ],
    'green': [
        (0.0, 0.0, 0.0),
        (0.5, 1.0, 1.0),
        (1.0, 0.0, 0.0),
    ],
    'blue': [
        (0.0, 0.0, 0.0),
        (0.5, 0.0, 0.0),
        (1.0, 0.0, 0.0),
    ],
}


DEFAULT_SEGMENTDATA_TWOSLOPE: LinearSegmentedColormapType = {
    'red': [
        (0.0, 0.0, 0.0),
        (0.5, 0.0, 0.0),
        (0.75, 1.0, 1.0),
        (1.0, 1.0, 1.0),
    ],
    'green': [
        (0.0, 0.0, 0.0),
        (0.25, 1.0, 1.0),
        (0.5, 0.0, 0.0),
        (0.75, 1.0, 1.0),
        (1.0, 0.0, 0.0),
    ],
    'blue': [
        (0.0, 1.0, 1.0),
        (0.25, 1.0, 1.0),
        (0.5, 0.0, 0.0),
        (1.0, 0.0, 0.0),
    ],
}

CmapNormType: TypeAlias = Union[
    matplotlib.colors.TwoSlopeNorm,
    matplotlib.colors.Normalize,
    matplotlib.colors.NoNorm,
]
MatplotlibSetupType: TypeAlias = tuple[
    matplotlib.pyplot.figure,
    matplotlib.pyplot.Axes,
    matplotlib.colors.Colormap,
    CmapNormType,
    np.ndarray,
    bool,
]


def setup_matplotlib(
        signal: np.ndarray,
        figsize: tuple[int, int],
        cmap: matplotlib.colors.Colormap | None = None,
        cmap_norm: matplotlib.colors.Normalize | str | None = None,
        cmap_segmentdata: LinearSegmentedColormapType | None = None,
        cval: np.ndarray | None = None,
        show_cbar: bool = False,
) -> MatplotlibSetupType:
    """Configure cmap.

    Parameters
    ----------
    signal: np.ndarray
        Time-step array.
    figsize: tuple[int, int]
        Figure size.
    cmap: matplotlib.colors.Colormap | None
        Color map for line color values. (default: None)
    cmap_norm: matplotlib.colors.Normalize | str | None
        Normalization for color values. (default: None)
    cmap_segmentdata: LinearSegmentedColormapType | None
        Color map segmentation to build color map. (default: None)
    cval: np.ndarray | None
        Line color values. (default: None)
    show_cbar: bool
        Shows color bar if True. (default: False)

    Returns
    -------
    MatplotlibSetupType
        Configures fig, ax, cmap, cmap_norm, cmap_segmentdata, cval, and show_cbar.
    """
    n = len(signal)

    fig = matplotlib.pyplot.figure(figsize=figsize)
    ax = fig.gca()

    if cval is None:
        cval = np.zeros(n)
        show_cbar = False

    cval_max = np.nanmax(np.abs(cval))
    cval_min = np.nanmin(cval).astype(float)

    if cmap_norm is None:
        if cval_max and cval_min < 0:
            cmap_norm = 'twoslope'
        elif cval_max:
            cmap_norm = 'normalize'
        else:
            cmap_norm = 'nonorm'

    if cmap is None:
        if cmap_segmentdata is None:
            if cmap_norm == 'twoslope':
                cmap_segmentdata = DEFAULT_SEGMENTDATA_TWOSLOPE
            else:
                cmap_segmentdata = DEFAULT_SEGMENTDATA

        cmap = matplotlib.colors.LinearSegmentedColormap(
            'line_cmap', segmentdata=cmap_segmentdata, N=512,
        )

    if cmap_norm == 'twoslope':
        cmap_norm = matplotlib.colors.TwoSlopeNorm(
            vcenter=0, vmin=-cval_max, vmax=cval_max,
        )
    elif cmap_norm == 'normalize':
        cmap_norm = matplotlib.colors.Normalize(
            vmin=cval_min, vmax=cval_max,
        )
    elif cmap_norm == 'nonorm':
        cmap_norm = matplotlib.colors.NoNorm()

    elif isinstance(cmap_norm, str):
        # pylint: disable=protected-access

        # to handle after https://github.com/pydata/xarray/pull/8030 is merged
        if (
            scale_class := matplotlib.scale._scale_mapping.get(cmap_norm, None)  # type: ignore
        ) is None:
            raise ValueError(f'cmap_norm string {cmap_norm} is not supported')

        norm_class = matplotlib.colors.make_norm_from_scale(scale_class)
        cmap_norm = norm_class(matplotlib.colors.Normalize)()

    return fig, ax, cmap, cmap_norm, cval, show_cbar


LinearSegmentedColormapType: TypeAlias = dict[
    Literal['red', 'green', 'blue', 'alpha'],
    Sequence[tuple[float, ...]],
]


DEFAULT_SEGMENTDATA: LinearSegmentedColormapType = {
    'red': [
        (0.0, 0.0, 0.0),
        (0.5, 1.0, 1.0),
        (1.0, 1.0, 1.0),
    ],
    'green': [
        (0.0, 0.0, 0.0),
        (0.5, 1.0, 1.0),
        (1.0, 0.0, 0.0),
    ],
    'blue': [
        (0.0, 0.0, 0.0),
        (0.5, 0.0, 0.0),
        (1.0, 0.0, 0.0),
    ],
}


DEFAULT_SEGMENTDATA_TWOSLOPE: LinearSegmentedColormapType = {
    'red': [
        (0.0, 0.0, 0.0),
        (0.5, 0.0, 0.0),
        (0.75, 1.0, 1.0),
        (1.0, 1.0, 1.0),
    ],
    'green': [
        (0.0, 0.0, 0.0),
        (0.25, 1.0, 1.0),
        (0.5, 0.0, 0.0),
        (0.75, 1.0, 1.0),
        (1.0, 0.0, 0.0),
    ],
    'blue': [
        (0.0, 1.0, 1.0),
        (0.25, 1.0, 1.0),
        (0.5, 0.0, 0.0),
        (1.0, 0.0, 0.0),
    ],
}

CmapNormType: TypeAlias = matplotlib.colors.TwoSlopeNorm | \
    matplotlib.colors.Normalize | \
    matplotlib.colors.NoNorm

MatplotlibSetupType: TypeAlias = tuple[
    matplotlib.pyplot.figure,
    matplotlib.pyplot.Axes,
    matplotlib.colors.Colormap,
    CmapNormType,
    np.ndarray,
    bool,
]

def draw_image_stimulus(
        image_stimulus: str | Path,
        origin: str = 'upper',
        show: bool = False,
) -> tuple[matplotlib.pyplot.figure, matplotlib.pyplot.Axes]:
    """Draw stimulus.

    Parameters
    ----------
    image_stimulus: str | Path
        Path to image stimulus.
    origin: str
        Origin how to draw the image.
    show: bool
        Boolean whether to show the image.

    Returns
    -------
    fig: matplotlib.pyplot.figure
    ax: matplotlib.pyplot.Axes
    """
    img = PIL.Image.open(image_stimulus)
    fig, ax = matplotlib.pyplot.subplots()
    ax.imshow(img, origin=origin)
    if show:
        matplotlib.pyplot.show()
    return fig, ax


def setup_matplotlib(
        signal: np.ndarray,
        figsize: tuple[int, int],
        cmap: matplotlib.colors.Colormap | None = None,
        cmap_norm: matplotlib.colors.Normalize | str | None = None,
        cmap_segmentdata: LinearSegmentedColormapType | None = None,
        cval: np.ndarray | None = None,
        show_cbar: bool = False,
) -> MatplotlibSetupType:
    """Configure cmap.

    Parameters
    ----------
    signal: np.ndarray
        Time-step array.
    figsize: tuple[int, int]
        Figure size.
    cmap: matplotlib.colors.Colormap | None
        Color map for line color values. (default: None)
    cmap_norm: matplotlib.colors.Normalize | str | None
        Normalization for color values. (default: None)
    cmap_segmentdata: LinearSegmentedColormapType | None
        Color map segmentation to build color map. (default: None)
    cval: np.ndarray | None
        Line color values. (default: None)
    show_cbar: bool
        Shows color bar if True. (default: False)

    Returns
    -------
    MatplotlibSetupType
        Configures fig, ax, cmap, cmap_norm, cmap_segmentdata, cval, and show_cbar.
    """
    n = len(signal)

    fig = matplotlib.pyplot.figure(figsize=figsize)
    ax = fig.gca()

    if cval is None:
        cval = np.zeros(n)
        show_cbar = False

    cval_max = np.nanmax(np.abs(cval))
    cval_min = np.nanmin(cval).astype(float)

    if cmap_norm is None:
        if cval_max and cval_min < 0:
            cmap_norm = 'twoslope'
        elif cval_max:
            cmap_norm = 'normalize'
        else:
            cmap_norm = 'nonorm'

    if cmap is None:
        if cmap_segmentdata is None:
            if cmap_norm == 'twoslope':
                cmap_segmentdata = DEFAULT_SEGMENTDATA_TWOSLOPE
            else:
                cmap_segmentdata = DEFAULT_SEGMENTDATA

        cmap = matplotlib.colors.LinearSegmentedColormap(
            'line_cmap', segmentdata=cmap_segmentdata, N=512,
        )

    if cmap_norm == 'twoslope':
        cmap_norm = matplotlib.colors.TwoSlopeNorm(
            vcenter=0, vmin=-cval_max, vmax=cval_max,
        )
    elif cmap_norm == 'normalize':
        cmap_norm = matplotlib.colors.Normalize(
            vmin=cval_min, vmax=cval_max,
        )
    elif cmap_norm == 'nonorm':
        cmap_norm = matplotlib.colors.NoNorm()

    elif isinstance(cmap_norm, str):
        # pylint: disable=protected-access

        # to handle after https://github.com/pydata/xarray/pull/8030 is merged
        if (
            scale_class := matplotlib.scale._scale_mapping.get(cmap_norm, None)  # type: ignore
        ) is None:
            raise ValueError(f'cmap_norm string {cmap_norm} is not supported')

        norm_class = matplotlib.colors.make_norm_from_scale(scale_class)
        cmap_norm = norm_class(matplotlib.colors.Normalize)()

    return fig, ax, cmap, cmap_norm, cval, show_cbar
