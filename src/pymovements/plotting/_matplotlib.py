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
"""Private helper functions for matplotlib."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Literal
from typing import Union

import matplotlib.colors
import matplotlib.pyplot
import numpy as np
import PIL.Image
from matplotlib.collections import LineCollection
from typing_extensions import TypeAlias


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


def _setup_matplotlib(
        x_signal: np.ndarray,
        y_signal: np.ndarray,
        figsize: tuple[int, int],
        cmap: matplotlib.colors.Colormap | None = None,
        cmap_norm: matplotlib.colors.Normalize | str | None = None,
        cmap_segmentdata: LinearSegmentedColormapType | None = None,
        cval: np.ndarray | None = None,
        show_cbar: bool = False,
        add_stimulus: bool = False,
        path_to_image_stimulus: str | None = None,
        stimulus_origin: str = 'upper',
        padding: float | None = None,
        pad_factor: float | None = 0.05,
) -> MatplotlibSetupType:
    """Configure cmap.

    Parameters
    ----------
    x_signal: np.ndarray
        Time-step array.
    y_signal: np.ndarray
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
    add_stimulus: bool
        Boolean value indicationg whether to plot the scanpath on the stimuls. (default: False)
    path_to_image_stimulus: str | None
        Path of the stimulus to be shown. (default: None)
    stimulus_origin: str
        Origin of stimuls to plot on the stimulus. (default: 'upper')
    padding: float | None
        Absolute padding value. If None it is inferred from pad_factor and limits. (default: None)
    pad_factor: float | None
        Relative padding factor to construct padding from value. (default: 0.05)

    Returns
    -------
    MatplotlibSetupType
        Configures fig, ax, cmap, cmap_norm, cmap_segmentdata, cval, and show_cbar.
    """
    n = len(x_signal)

    fig = matplotlib.pyplot.figure(figsize=figsize)
    ax = fig.gca()

    if add_stimulus:
        img = PIL.Image.open(path_to_image_stimulus)
        ax.imshow(img, origin=stimulus_origin, extent=None)
    else:
        if padding is None:
            x_pad = (np.nanmax(x_signal) - np.nanmin(x_signal)) * pad_factor
            y_pad = (np.nanmax(y_signal) - np.nanmin(y_signal)) * pad_factor
        else:
            x_pad = padding
            y_pad = padding

        ax.set_xlim(np.nanmin(x_signal) - x_pad, np.nanmax(x_signal) + x_pad)
        ax.set_ylim(np.nanmin(y_signal) - y_pad, np.nanmax(y_signal) + y_pad)
        ax.invert_yaxis()

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


def _draw_line_data(
        x_signal: np.ndarray,
        y_signal: np.ndarray,
        ax: matplotlib.pyplot.Axes,
        cmap: matplotlib.colors.Colormap | None = None,
        cmap_norm: matplotlib.colors.Normalize | str | None = None,
        cval: np.ndarray | None = None,
) -> matplotlib.pyplot.Axes:
    """Draw line data.

    Parameters
    ----------
    x_signal: np.ndarray
        Data to be plotted.
    y_signal: np.ndarray
        Data to be plotted.
    ax: matplotlib.pyplot.Axes
        Matplotlib axes.
    cmap: matplotlib.colors.Colormap | None
        Color map for line color values. (default: None)
    cmap_norm: matplotlib.colors.Normalize | str | None
        Normalization for color values. (default: None)
    cval: np.ndarray | None
        Line color values. (default: None)

    Returns
    -------
    matplotlib.pyplot.Axes
        Axes with added line data.

    """
    # Create a set of line segments so that we can color them individually
    # This creates the points as a N x 1 x 2 array so that we can stack points
    # together easily to get the segments. The segments array for line collection
    # needs to be (numlines) x (points per line) x 2 (for x and y)
    points = np.array([x_signal, y_signal]).T.reshape((-1, 1, 2))
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    # Create a continuous norm to map from data points to colors
    line_collection = LineCollection(segments, cmap=cmap, norm=cmap_norm)
    # Set the values used for colormapping
    line_collection.set_array(cval)
    line_collection.set_linewidth(2)
    line = ax.add_collection(line_collection)
    return line
