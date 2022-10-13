from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import matplotlib
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection


default_segmentdata = {
    'red':   [
        [0.0,  0.0, 0.0],
        [0.5,  1.0, 1.0],
        [1.0,  1.0, 1.0],
    ],
    'green': [
        [0.0,  0.0, 0.0],
        [0.5,  1.0, 1.0],
        [1.0,  0.0, 0.0],
    ],
    'blue':  [
        [0.0,  0.0, 0.0],
        [0.5,  0.0, 0.0],
        [1.0,  0.0, 0.0],
    ],
}

default_segmentdata_twoslope = {
    'red':   [
        [0.0,  0.0, 0.0],
        [0.5,  0.0, 0.0],
        [0.75, 1.0, 1.0],
        [1.0,  1.0, 1.0],
    ],
    'green': [
        [0.0,  0.0, 0.0],
        [0.25, 1.0, 1.0],
        [0.5,  0.0, 0.0],
        [0.75, 1.0, 1.0],
        [1.0,  0.0, 0.0],
    ],
    'blue':  [
        [0.0,  1.0, 1.0],
        [0.25, 1.0, 1.0],
        [0.5,  0.0, 0.0],
        [1.0,  0.0, 0.0],
    ],
}


def traceplot(
    x: np.array,
    y: np.array,
    cval: Optional[np.array] = None,
    cmap: Optional[colors.Colormap] = None,
    cmap_norm: Optional[colors.Normalize] = None,
    cmap_segmentdata: Optional[Dict[str, List[List[float]]]] = None,
    cbar_label: Optional[str] = None,
    show_cbar: bool = False,
    padding: Optional[float] = None,
    pad_factor: float = 0.05,
    figsize: Tuple[int, int] = (15, 5),
    title: Optional[str] = None,
    savepath: Optional[str] = None,
    show: bool = True,
) -> None:

    if len(x) != len(y):
        raise ValueError(
            'x and y do not share same length '
            f'({len(x)} != {len(y)})',
        )
    n = len(x)

    fig = plt.figure(figsize=figsize)
    ax = fig.gca()

    if cval is None:
        cval = np.zeros(n)
        show_cbar = False

    if cmap_norm is None:
        cval_max = np.nanmax(np.abs(cval))
        cval_min = np.nanmin(cval)

        if cval_max and cval_min < 0:
            cmap_norm = 'twoslope'
        elif cval_max:
            cmap_norm = 'normalize'
        else:
            cmap_norm = 'nonorm'

    if cmap is None:
        if cmap_segmentdata is None:
            if cmap_norm == 'twoslope':
                cmap_segmentdata = default_segmentdata_twoslope
            else:
                cmap_segmentdata = default_segmentdata

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

    # Create a set of line segments so that we can color them individually
    # This creates the points as a N x 1 x 2 array so that we can stack points
    # together easily to get the segments. The segments array for line collection
    # needs to be (numlines) x (points per line) x 2 (for x and y)
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Create a continuous norm to map from data points to colors
    line_collection = LineCollection(segments, cmap=cmap, norm=cmap_norm)
    # Set the values used for colormapping
    line_collection.set_array(cval)
    line_collection.set_linewidth(2)
    line = ax.add_collection(line_collection)

    if padding is None:
        x_pad = (np.nanmax(x) - np.nanmin(x)) * pad_factor
        y_pad = (np.nanmax(y) - np.nanmin(y)) * pad_factor
    else:
        x_pad = padding
        y_pad = padding

    ax.set_xlim(np.nanmin(x) - x_pad, np.nanmax(x) + x_pad)
    ax.set_ylim(np.nanmin(y) - y_pad, np.nanmax(y) + y_pad)

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


def tsplot(
    arr: np.array,
    channel_names: Optional[List[str]] = None,
    channel_axis: int = 0,
    xlabel: Optional[str] = None,
    sample_axis: int = 1,
    events: Optional[np.array] = None,
    share_y: bool = True,
    zero_centered_yaxis: bool = True,
    line_color: str = 'k',  # TODO: use correct color type
    line_width: int = 1,
    rotate_ylabels: bool = True,
    show_grid: bool = True,
    show_yticks: bool = True,
    figsize: Tuple[int, int] = (15, 5),
    title: Optional[str] = None,
    savepath: Optional[str] = None,
    show: bool = True,
) -> None:

    if arr.ndim == 1:
        arr = np.expand_dims(arr, axis=0)
    elif arr.ndim == 2:
        pass
    else:
        raise ValueError(arr.shape)

    if channel_axis != 0:
        raise NotImplementedError()
    if sample_axis != 1:
        raise NotImplementedError()

    n_channels = arr.shape[channel_axis]
    n_samples = arr.shape[sample_axis]

    # determine number of subplots and height ratios for events
    if events is not None:
        raise NotImplementedError()
        # n_subplots = n_channels + len(events)
        # height_ratios = height_ratios + [1 * n_event_types]
    else:
        n_subplots = n_channels
        height_ratios = [1] * n_channels

    fig, axs = plt.subplots(
        nrows=n_subplots,
        sharex=True, sharey=share_y,
        figsize=figsize,
        gridspec_kw={
            'hspace': 0,
            'height_ratios': height_ratios,
        },
    )

    t = np.arange(n_samples)
    xlims = t.min(), t.max()

    # set ylims to have zero centered y-axis (for all axes)
    if share_y and zero_centered_yaxis:
        ylim_abs = np.nanmax(np.abs(arr))
        # TODO: factor 1.1 out as padding argument
        ylims = -ylim_abs * 1.1, ylim_abs * 1.1

    for channel_id in range(n_channels):
        ax = axs[channel_id]
        x_channel = arr[channel_id, :]

        # set ylims to have zero centered y-axis (for single axis)
        # TODO: fix zerp_centered_yaxis undefined
        # if not share_y and zerp_centered_yaxis:
        #     ylim_abs = np.nanmax(np.abs(x_channel))
        #     ylims = -ylim_abs * 1.1, ylim_abs * 1.1

        ax.plot(t, x_channel, color=line_color, linewidth=line_width)

        '''
        n_attr = 1 #len(attributions)
        attribution_dict = {' ': attributions}
        for i_attr, (attr_name, attr_vals) in enumerate(attribution_dict.items()):
            y_seg = (ylims[1] - ylims[0]) / n_attr
            y_bot = i_attr * y_seg + ylims[0]

            y_top = y_bot + y_seg
            y_mid = y_bot + y_seg / 2

            extent = xlims[0], xlims[1], y_bot, y_top

            attr_val_max = max(abs(attr_vals.min()), abs(attr_vals.max()))
            colornorm = colors.TwoSlopeNorm(
                vmin=-attr_val_max,
                vcenter=0,
                vmax=attr_val_max,
            )
            im = ax.imshow(X=np.expand_dims(attr_vals[:, channel_id], 0),
                           norm=colornorm,
                           cmap=plt.cm.coolwarm, interpolation='nearest',
                           extent=extent, alpha=1, aspect='auto')

            #if feature_importance_names is not None:
            #    ax.text(-10, y_mid, feature_importance_names[i_fimp],
            #            ha='right', va='center')

        #fig.colorbar(im, ax=ax, orientation='vertical')
        '''

        ax.set_xlim(xlims)
        ax.set_ylim(ylims)

        # TODO: add setting for minor and major
        ax.grid(show_grid, which='major')
        ax.grid(show_grid, which='minor')

        ax.tick_params(
            which='both', direction='out',
            length=4, width=1, colors='k',
            grid_color='#999999', grid_alpha=0.5,
        )

        if show_yticks:
            # from matplotlib.ticker import AutoMinorLocator
            # ax.yaxis.set_minor_locator(AutoMinorLocator())
            plt.setp(ax.get_yticklabels(), visible=True)
        else:
            ax.set_yticks([])
            plt.setp(ax.get_yticklabels(), visible=False)

        # TODO: find out why this is needed
        params = {'mathtext.default': 'regular'}
        plt.rcParams.update(params)

        # set channel names as y-axis labels
        if channel_names:
            if rotate_ylabels:
                ax.set_ylabel(
                    channel_names[channel_id],
                    rotation='horizontal',
                    ha='right', va='center',
                )
            else:
                ax.set_ylabel(channel_names[channel_id])

        # TODO: find out why this is needed
        # if False: # events is not None or channel_id != n_channels - 1:
        #     raise NotImplementedError()
        #     plt.setp(ax.get_xticklabels(), visible=False)

    # TODO fix: inputs not defined
    # if events is not None:
    #     raise NotImplementedError()
    #     ax = axs[-1]
    #     t = list(range(len(inputs)))

    #     height = 1
    #     padding = 0.1

    #     for event_id, (event_name, event_data) in enumerate(events.items()):
    #         y_top = (n_event_types - event_id) * height
    #         y_bot = y_top - height
    #         y_mid = y_top - height / 2

    #         y_top_padded = y_top - padding
    #         y_bot_padded = y_bot + padding
    #         height_padded = height - 2 * padding

    #         event_color = event_colors[event_id]
    #         for _, event in event_data.iterrows():
    #             event_xy = event.start, y_bot_padded
    #             event_length = event.end - event.start

    #             patch = Rectangle(
    #                 xy=event_xy,
    #                 width=event_length,
    #                 height=height_padded,
    #                 color=event_color,
    #             )
    #             ax.add_patch(patch)

    #         ax.text(-10, y_mid, event_name, ha='right', va='center')

    #     ax.set_yticks([])
    #     ax.xaxis.grid(True, which='major')
    #     ax.xaxis.grid(True, which='minor')
    #     ax.yaxis.grid(False, which='major')
    #     ax.yaxis.grid(False, which='minor')

    #     ax.set_xlim(xlims)
    #     ax.set_ylim(0, n_event_types)

    # print x label on last used (bottom) axis
    ax.set_xlabel(xlabel)

    # TODO implement correctly
    # if False: #show_cbar: # and (subfig_id + 1 == n_subfigs):
    #    fig.subplots_adjust(right=0.8)
    #    # put colorbar at desire position
    #    cbar_ax = fig.add_axes([0.85, 0.1, 0.025, 0.8])
    #    cbar = fig.colorbar(im, cax=cbar_ax)
    #    cbar.ax.set_ylabel(cbar_label, rotation=90)

    #    '''
    #    from mpl_toolkits.axes_grid1 import make_axes_locatable
    #    #divider = make_axes_locatable(ax)
    #    #cax = divider.append_axes('right', size='5%', pad=0.05)
    #    cax = None
    #    fig.colorbar(im, cax=cax, orientation='vertical')
    #    '''

    axs[0].set_title(title)

    if savepath is not None:
        fig.savefig(savepath)

    if show:
        plt.show()
    plt.close(fig)
