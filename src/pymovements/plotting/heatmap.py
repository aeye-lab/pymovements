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
Heatmap module.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors

from pymovements.gaze import GazeDataFrame


def heatmap(
    gaze: GazeDataFrame,
    position_columns: tuple[str, str] = ('x_pix', 'y_pix'),
    gridsize=(10, 10),
    cmap: colors.Colormap | str = 'jet',
    interpolation: str = 'gaussian',
    origin: str = 'lower',
    figsize: tuple[float, float] = (15, 10),
    cbar_label: str | None = None,
    show_cbar: bool = True,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    show: bool = True,
    savepath: str | None = None,
) -> plt.Figure:
    """Plot a heatmap of gaze data.

    The heatmap displays the distribution of gaze positions across the experiment screen,
    for a given GazeDataFrame object.
    The color values indicate the time spent at each position in seconds.

    Parameters
    ----------
    gaze : GazeDataFrame
        A GazeDataFrame object.
    position_columns : tuple[str, str], optional
        The column names of the x and y position data
    gridsize : tuple[int, int], optional
        The number of bins in the x and y dimensions.
    cmap : colors.Colormap | str, optional
        The colormap to use
    interpolation : str, optional
        The interpolation method to use for plotting the heatmap.
        See matplotlib.pyplot.imshow for more information on available methods
        for interpolation. By default, 'gaussian' is used.
    origin : str, optional
        origin of y-axis, valid values are 'lower' or 'upper'
    figsize : tuple[float, float], optional
        Figure size
    cbar_label : str | None, optional
        Label for the colorbar
    show_cbar : bool, optional
        Whether to show the colorbar.
    title : str | None, optional
        Figure title
    xlabel : str | None, optional
        x-axis label
    ylabel : str | None, optional
        y-axis label
    show : bool, optional
        Whether to show the plot
    savepath : str | None, optional
        If provided, the figure will be saved to this path

    Raises
    ------
    ValueError
        If the position columns are not in pixels or degrees
    ValueError
        If the experiment property of the GazeDataFrame is None
    Returns
    -------
    plt.Figure
        The heatmap figure.
    """

    # Extract x and y positions from the gaze dataframe
    x = gaze.frame[position_columns[0]]
    y = gaze.frame[position_columns[1]]

    # Check if experiment properties are available
    if not gaze.experiment:
        raise ValueError(
            'Experiment property of GazeDataFrame is None. '
            'GazeDataFrame must be associated with an experiment.',
        )

    # Get experiment screen properties
    screen = gaze.experiment.screen

    # Use screen properties to define the grid or degrees of visual angle
    if 'pix' in position_columns[0] and 'pix' in position_columns[1]:
        xmin, xmax = 0, screen.width_px
        ymin, ymax = 0, screen.height_px
    else:
        xmin, xmax = int(screen.x_min_dva), int(screen.x_max_dva)
        ymin, ymax = int(screen.y_min_dva), int(screen.y_max_dva)

    # Define the grid and bin the gaze data
    x_bins = np.linspace(xmin, xmax, num=gridsize[0]).astype(int)
    y_bins = np.linspace(ymin, ymax, num=gridsize[1]).astype(int)

    # Bin the gaze data
    heatmap_value, x_edges, y_edges = np.histogram2d(x, y, bins=[x_bins, y_bins])

    # Transpose to match the orientation of the screen
    heatmap_value = heatmap_value.T

    # Convert heatmap values from sample count to seconds
    heatmap_value /= gaze.experiment.sampling_rate

    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)

    # Plot the heatmap
    heatmap_plot = ax.imshow(
        heatmap_value,
        cmap=cmap,
        origin=origin,
        interpolation=interpolation,
        extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
    )

    # Set the plot title and axis labels
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)

    # Add a color bar to the plot
    if show_cbar:
        cbar = fig.colorbar(heatmap_plot, ax=ax)
        if cbar_label:
            cbar.set_label(cbar_label)

    # Show or save the plot
    if savepath:
        plt.savefig(savepath)
    if show:
        plt.show()

    return fig
