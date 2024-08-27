# Copyright (c) 2022-2024 The pymovements Project Authors
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
"""Provides the scanpath plotting function."""
from __future__ import annotations

import math
import sys

import matplotlib.colors
import matplotlib.pyplot as plt
import matplotlib.scale
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle

from pymovements.events import EventDataFrame
from pymovements.gaze import GazeDataFrame
from pymovements.utils.plotting import LinearSegmentedColormapType
from pymovements.utils.plotting import setup_matplotlib


# This is really a dirty workaround to use the Agg backend if runnning pytest.
# This is needed as Windows workers on GitHub fail randomly with other backends.
# Unfortunately the Agg module cannot show plots in jupyter notebooks.
if 'pytest' in sys.modules:  # pragma: no cover
    matplotlib.use('Agg')


def scanpathplot(
        events: EventDataFrame,
        gaze: GazeDataFrame | None = None,
        position_column: str = 'location',
        cval: np.ndarray | None = None,  # pragma: no cover
        cmap: matplotlib.colors.Colormap | None = None,
        cmap_norm: matplotlib.colors.Normalize | str | None = None,
        cmap_segmentdata: LinearSegmentedColormapType | None = None,
        cbar_label: str | None = None,
        show_cbar: bool = False,
        padding: float | None = None,
        pad_factor: float | None = 0.05,
        figsize: tuple[int, int] = (15, 5),
        title: str | None = None,
        savepath: str | None = None,
        show: bool = True,
        color: str = 'blue',
        alpha: float = 0.5,
        add_traceplot: bool = False,
        gaze_position_column: str = 'pixel',
) -> None:
    """Plot scanpath from positional data.

    Parameters
    ----------
    events: EventDataFrame
        The EventDataFrame to plot.
    gaze: GazeDataFrame | None
        Optional Gaze Dataframe. (default: None)
    position_column: str
        The column name of the x and y position data (default: 'pixel')
    cval: np.ndarray | None
        Line color values. (default: None)
    cmap: matplotlib.colors.Colormap | None
        Color map for line color values. (default: None)
    cmap_norm: matplotlib.colors.Normalize | str | None
        Normalization for color values. (default: None)
    cmap_segmentdata: LinearSegmentedColormapType | None
        Color map segmentation to build color map. (default: None)
    cbar_label: str | None
        String label for color bar. (default: None)
    show_cbar: bool
        Shows color bar if True. (default: False)
    padding: float | None
        Absolute padding value.
        If None it is inferred from pad_factor and limits. (default: None)
    pad_factor: float | None
        Relative padding factor to construct padding value if not given. (default: None)
    figsize: tuple[int, int]
        Figure size. (default: (15, 5))
    title: str | None
        Set figure title. (default: None)
    savepath: str | None
        If given, figure will be saved to this path. (default: None)
    show: bool
        If True, figure will be shown. (default: True)
    color: str
        Color of fixations. (default: 'blue')
    alpha: float
        Alpha value of scanpath. (default: 0.5)
    add_traceplot: bool
        Boolean value indicating whether to add traceplot to the scanpath plot. (default: False)
    gaze_position_column: str
        Position column in the gaze dataframe. (default: 'pixel')

    Raises
    ------
    ValueError
        If length of x and y coordinates do not match or if ``cmap_norm`` is unknown.

    """
    x_signal = events.frame[position_column].list.get(0)
    y_signal = events.frame[position_column].list.get(1)

    fig, ax, cmap, cmap_norm, cval, show_cbar = setup_matplotlib(
        x_signal,
        figsize,
        cmap,
        cmap_norm,
        cmap_segmentdata,
        cval,
        show_cbar,
    )

    for row in events.frame.iter_rows(named=True):
        fixation = Circle(
            row[position_column],
            math.sqrt(row['duration']),
            color=color,
            fill=True,
            alpha=alpha,
            zorder=10,
        )
        ax.add_patch(fixation)

    if add_traceplot:
        assert gaze
        gaze_x_signal = gaze.frame[gaze_position_column].list.get(0)
        gaze_y_signal = gaze.frame[gaze_position_column].list.get(1)
        points = np.array([gaze_x_signal, gaze_y_signal]).T.reshape((-1, 1, 2))
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # Create a continuous norm to map from data points to colors
        line_collection = LineCollection(segments, cmap=cmap, norm=cmap_norm)
        # Set the values used for colormapping
        line_collection.set_array(cval)
        line_collection.set_linewidth(2)
        line = ax.add_collection(line_collection)
        if show_cbar:
            # sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=cmap_norm)
            # sm.set_array(cval)
            fig.colorbar(line, label=cbar_label, ax=ax)

    if padding is None:
        x_pad = (np.nanmax(x_signal) - np.nanmin(x_signal)) * pad_factor
        y_pad = (np.nanmax(y_signal) - np.nanmin(y_signal)) * pad_factor
    else:
        x_pad = padding
        y_pad = padding

    ax.set_xlim(np.nanmin(x_signal) - x_pad, np.nanmax(x_signal) + x_pad)
    ax.set_ylim(np.nanmin(y_signal) - y_pad, np.nanmax(y_signal) + y_pad)

    if title:
        ax.set_title(title)

    if savepath is not None:
        fig.savefig(savepath)

    if show:
        plt.show()
    plt.close(fig)