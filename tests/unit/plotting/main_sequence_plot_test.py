# Copyright (c) 2023 The pymovements Project Authors
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
"""Test main_sequence_plot."""
from unittest.mock import Mock

import numpy as np
import polars as pl
import pytest
from matplotlib import pyplot as plt

from pymovements.events import EventDataFrame
from pymovements.plotting.main_sequence_plot import main_sequence_plot


@pytest.mark.parametrize(
    ('input_df', 'show', 'color', 'marker', 'alpha', 'size'),
    [
        pytest.param(
            EventDataFrame(
                pl.DataFrame(
                    {
                        'amplitude': [1.0, 1.0, 2.0, 2.0, 3.0, 4.0],
                        'peak_velocity': [10.0, 11.0, 12.0, 11.0, 13.0, 13.0],
                        'name': ['saccade' for _ in range(6)],

                    },
                ),
            ),
            True,
            'blue',
            'x',
            0.6,
            30,
            id='show_plot',
        ),
    ],
)
def test_main_sequence_plot_show_plot(input_df, show, monkeypatch, color, marker, alpha, size):
    mock_show = Mock()
    mock_scatter = Mock()

    monkeypatch.setattr(plt, 'show', mock_show)
    monkeypatch.setattr(plt, 'scatter', mock_scatter)

    main_sequence_plot(
        input_df,
        show=show,
        color=color,
        marker=marker,
        alpha=alpha,
        marker_size=size,
    )
    plt.close()

    mock_scatter.assert_called_with(
        [1.0, 1.0, 2.0, 2.0, 3.0, 4.0],
        [10.0, 11.0, 12.0, 11.0, 13.0, 13.0],
        color='blue',
        alpha=0.6,
        s=30,
        marker='x',
    )

    mock_show.assert_called_once()


@pytest.mark.parametrize(
    'input_df',
    [
        pytest.param(
            EventDataFrame(
                pl.DataFrame(
                    {
                        'amplitude': [1.0, 1.0, 2.0, 2.0, 3.0, 4.0],
                        'peak_velocity': [10.0, 11.0, 12.0, 11.0, 13.0, 13.0],
                        'name': ['saccade' for _ in range(5)] + ['fixation'],

                    },
                ),
            ),
            id='filter_out_fixations',
        ),
    ],
)
def test_main_sequence_plot_filter_out_fixations(input_df, monkeypatch):
    mock_scatter = Mock()

    monkeypatch.setattr(plt, 'scatter', mock_scatter)

    main_sequence_plot(input_df, show=False)
    plt.close()

    mock_scatter.assert_called_with(
        [1.0, 1.0, 2.0, 2.0, 3.0],
        [10.0, 11.0, 12.0, 11.0, 13.0],
        color='purple',
        alpha=0.5,
        s=25,
        marker='o',
    )


@pytest.mark.parametrize(
    'input_df',
    [
        pytest.param(
            EventDataFrame(
                pl.DataFrame(
                    {
                        'amplitude': np.arange(100),
                        'peak_velocity': np.linspace(10, 50, num=100),
                        'name': ['saccade' for _ in range(100)],

                    },
                ),
            ),
            id='save_path',
        ),
    ],
)
def test_main_sequence_plot_save_path(input_df, monkeypatch):
    mock_function = Mock()
    monkeypatch.setattr(plt.Figure, 'savefig', mock_function)
    main_sequence_plot(input_df, show=False, savepath='mock')
    plt.close()
    mock_function.assert_called_once()


@pytest.mark.parametrize(
    ('input_df', 'show'),
    [
        pytest.param(
            EventDataFrame(
                pl.DataFrame(
                    {
                        'amplitude': np.arange(100),
                        'peak_velocity': np.linspace(10, 50, num=100),
                        'name': ['saccade' for _ in range(100)],
                    },
                ),
            ),
            False,
            id='do_not_show_plot',
        ),
    ],
)
def test_main_sequence_plot_not_show(input_df, show, monkeypatch):
    mock_function = Mock()
    monkeypatch.setattr(plt, 'show', mock_function)
    main_sequence_plot(input_df, show=show)
    plt.close()
    mock_function.assert_not_called()


@pytest.mark.parametrize(
    ('input_df', 'title'),
    [
        pytest.param(
            EventDataFrame(
                pl.DataFrame(
                    {
                        'amplitude': np.arange(100),
                        'peak_velocity': np.linspace(10, 50, num=100),
                        'name': ['saccade' for _ in range(100)],
                    },
                ),
            ),
            'foo',
            id='do_not_show_plot',
        ),
    ],
)
def test_main_sequence_plot_set_title(input_df, title, monkeypatch):
    mock_function = Mock()
    monkeypatch.setattr(plt, 'title', mock_function)
    main_sequence_plot(input_df, title=title)
    plt.close()


@pytest.mark.parametrize(
    ('input_df', 'expected_error', 'error_msg'),
    [
        pytest.param(
            EventDataFrame(
                pl.DataFrame(
                    {
                        'peak_velocity': np.linspace(10, 50, num=100),
                        'name': ['saccade' for _ in range(100)],

                    },
                ),
            ),
            KeyError,
            'The input dataframe you provided does not contain '
            'the saccade amplitudes which are needed to create '
            'the main sequence plot. ',
            id='amplitude_missing',
        ),
        pytest.param(
            EventDataFrame(
                pl.DataFrame(
                    {
                        'amplitude': np.arange(100),
                        'name': ['saccade' for _ in range(100)],
                    },
                ),
            ),
            KeyError,
            'The input dataframe you provided does not contain '
            'the saccade peak velocities which are needed to create '
            'the main sequence plot. ',
            id='peak_velocity_missing',
        ),
        pytest.param(
            EventDataFrame(
                pl.DataFrame(
                    {
                        'amplitude': [1.0, 1.0],
                        'peak_velocity': [10.0, 11.0],
                        'name': ['fixation', 'fixation'],
                    },
                ),
            ),
            ValueError,
            'There are no saccades in the event dataframe. '
            'Please make sure you ran a saccade detection algorithm. '
            'The event name should be stored in a colum called "name".',
            id='no_saccades_in_event_df',
        ),
    ],
)
def test_main_sequence_plot_error(input_df, expected_error, error_msg):
    with pytest.raises(expected_error) as actual_error:
        main_sequence_plot(input_df)

    msg, = actual_error.value.args

    assert msg == error_msg
