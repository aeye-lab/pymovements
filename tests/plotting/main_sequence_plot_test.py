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

from pymovements.plotting.main_sequence_plot import main_sequence_plot


@pytest.mark.parametrize(
    ('input_df', 'show'),
    [
        pytest.param(
            pl.DataFrame(
                {
                    'amplitude': [1.0, 1.0, 2.0, 2.0, 3.0, 4.0],
                    'peak_velocity': [10.0, 11.0, 12.0, 11.0, 13.0, 13.0],
                },
            ),
            True,
            id='show_plot'
        ),
    ],
)
def test_main_sequence_plot_show_plot(input_df, show, monkeypatch):
    mock = Mock()
    monkeypatch.setattr(plt, 'show', mock)
    main_sequence_plot(input_df, show=show)
    mock.assert_called_once()


@pytest.mark.parametrize(
    ('input_df', 'show'),
    [
        pytest.param(
            pl.DataFrame(
                {
                    'amplitude': [1.0, 1.0, 2.0, 2.0, 3.0, 4.0],
                    'peak_velocity': [10.0, 11.0, 12.0, 11.0, 13.0, 13.0],
                },
            ),
            False,
            id='do_not_show_plot',
        ),
    ],
)
def test_main_sequence_plot_not_show_plot(input_df, show, monkeypatch):
    mock = Mock()
    monkeypatch.setattr(plt, 'show', mock)
    main_sequence_plot(input_df, show=show)
    mock.assert_not_called()


@pytest.mark.parametrize(
    ('input_df', 'show'),
    [
        pytest.param(
            pl.from_dict(
                {
                    'subject_id': np.ones(10),
                    'time': np.arange(10),
                    'x_pos': np.concatenate([np.ones(5), np.zeros(5)]),
                    'y_pos': np.concatenate([np.zeros(5), np.ones(5)]),
                    #'x_vel': np.concatenate([np.ones(5), np.zeros(5)]),
                    #'y_vel': np.zeros(10),
                },
                schema={
                    'subject_id': pl.Int64,
                    'time': pl.Int64,
                    'x_pos': pl.Float64,
                    'y_pos': pl.Float64,
                   # 'x_vel': pl.Float64,
                    #'y_vel': pl.Float64,
                },
            ),
            False,
            id='do_not_show_plot',
        ),
    ],
)
def test_main_sequence_plot_not_show_plot(input_df, show, monkeypatch):
    mock = Mock()
    monkeypatch.setattr(plt, 'show', mock)
    main_sequence_plot(input_df, show=show)
    mock.assert_not_called()
