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
"""
This module holds the traceplot.
"""
from __future__ import annotations

import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.collections import LineCollection

from pymovements.gaze.gaze_dataframe import GazeDataFrame

# This is really a dirty workaround to use the Agg backend if runnning pytest.
# This is needed as Windows workers on GitHub fail randomly with other backends.
# Unfortunately the Agg module cannot show plots in jupyter notebooks.
if 'pytest' in sys.modules:  # pragma: no cover
    matplotlib.use('Agg')

DEFAULT_SEGMENTDATA = {
    'red': [
        [0.0, 0.0, 0.0],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, 1.0],
    ],
    'green': [
        [0.0, 0.0, 0.0],
        [0.5, 1.0, 1.0],
        [1.0, 0.0, 0.0],
    ],
    'blue': [
        [0.0, 0.0, 0.0],
        [0.5, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ],
}


DEFAULT_SEGMENTDATA_TWOSLOPE = {
    'red': [
        [0.0, 0.0, 0.0],
        [0.5, 0.0, 0.0],
        [0.75, 1.0, 1.0],
        [1.0, 1.0, 1.0],
    ],
    'green': [
        [0.0, 0.0, 0.0],
        [0.25, 1.0, 1.0],
        [0.5, 0.0, 0.0],
        [0.75, 1.0, 1.0],
        [1.0, 0.0, 0.0],
    ],
    'blue': [
        [0.0, 1.0, 1.0],
        [0.25, 1.0, 1.0],
        [0.5, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ],
}


def traceplot(
        gaze: GazeDataFrame,
        x: str,
        y: str,
        cval: np.ndarray | None = None,  # pragma: no cover
        cmap: colors.Colormap | None = None,
        cmap_norm: colors.Normalize | str | None = None,
        cmap_segmentdata: dict[str, list[list[float]]] | None = None,
        cbar_label: str | None = None,
        show_cbar: bool = False,
        padding: float | None = None,
        pad_factor: float | None = 0.05,
        figsize: tuple[int, int] = (15, 5),
        title: str | None = None,
        savepath: str | None = None,
        show: bool = True,
) -> None:
    """
    Plot eye gaze trace from positional data.

    Parameters
    ----------
    gaze: GazeDataFrame
        The GazeDataFrame to plot.
    x: str
        Column name of x-coordinates
    y: str
        Column name of y-coordinates
    cval: np.ndarray
        line color values.
    cmap: matplotlib.colors.Colormap, optional
        color map for line color values
    cmap_norm: matplotlib.colors.Normalize, str, optional
        normalization for color values.
    cmap_segmentdata: dict, optional
        color map segmentation to build color map
    cbar_label: str, optional
        string label for color bar
    show_cbar: bool
        Shows color bar if True.
    padding: float, optional
        Absolute padding value. If None it is inferred from pad_factor and limits.
    pad_factor: float, optional
        Relative padding factor to construct padding value if not given.
    figsize: tuple
        Figure size.
    title: str, optional
        Figure title.
    savepath: str, optional
        If given, figure will be saved to this path.
    show: bool
        If True, figure will be shown.

    Raises
    ------
    ValueError
        If length of x and y coordinates do not match or if ``cmap_norm`` is unknown.

    """
    x_signal = gaze.frame[x]
    y_signal = gaze.frame[y]

    n = len(x_signal)

    fig = plt.figure(figsize=figsize)
    ax = fig.gca()

    if cval is None:
        cval = np.zeros(n)
        show_cbar = False

    cval_max = np.nanmax(np.abs(cval))
    cval_min = np.nanmin(cval)

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
        scale_class = matplotlib.scale._scale_mapping.get(cmap_norm, None)

        if scale_class is None:
            raise ValueError(f'cmap_norm string {cmap_norm} is not supported')

        norm_class = matplotlib.colors.make_norm_from_scale(scale_class)
        cmap_norm = norm_class(matplotlib.colors.Normalize)()

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

    if padding is None:
        x_pad = (np.nanmax(x_signal) - np.nanmin(x_signal)) * pad_factor
        y_pad = (np.nanmax(y_signal) - np.nanmin(y_signal)) * pad_factor
    else:
        x_pad = padding
        y_pad = padding

    ax.set_xlim(np.nanmin(x_signal) - x_pad, np.nanmax(x_signal) + x_pad)
    ax.set_ylim(np.nanmin(y_signal) - y_pad, np.nanmax(y_signal) + y_pad)

    if show_cbar:
        # sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=cmap_norm)
        # sm.set_array(cval)
        fig.colorbar(line, label=cbar_label, ax=ax)

    ax.set_title(title)

    if savepath is not None:
        fig.savefig(savepath)

    if show:
        plt.show()
    plt.close(fig)
