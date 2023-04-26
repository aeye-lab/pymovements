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
"""Test tsplot."""
from unittest.mock import Mock

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib import figure

from pymovements.plotting import tsplot


@pytest.fixture(name='gaze')
def gaze_fixture():
    x = np.arange(-100, 100)
    y = np.arange(-100, 100)
    arr = np.column_stack((x, y)).transpose()

    return arr


@pytest.mark.parametrize(
    'kwargs',
    [
        pytest.param({}, id='no_kwargs'),
        pytest.param({'share_y': False}, id='share_y_false'),
        pytest.param({'show_yticks': False}, id='show_yticks_false'),
        pytest.param({'channel_names': ['foo', 'bar']}, id='channel_names'),
        pytest.param(
            {'channel_names': ['foo', 'bar'], 'rotate_ylabels': False},
            id='channel_names_no_rotate',
        ),
    ],
)
def test_tsplot_show(arr, kwargs, monkeypatch):
    mock = Mock()
    monkeypatch.setattr(plt, 'show', mock)
    tsplot(arr, **kwargs)
    plt.close()
    mock.assert_called_once()


def test_tsplot_1d(monkeypatch):
    mock = Mock()
    monkeypatch.setattr(plt, 'show', mock)
    tsplot(np.arange(-100, 100))
    plt.close()
    mock.assert_called_once()


def test_tsplot_noshow(arr, monkeypatch):
    mock = Mock()
    monkeypatch.setattr(plt, 'show', mock)
    tsplot(arr, show=False)
    plt.close()
    mock.assert_not_called()


def test_tsplot_save(arr, monkeypatch, tmp_path):
    mock = Mock()
    monkeypatch.setattr(figure.Figure, 'savefig', mock)
    tsplot(arr, show=False, savepath=str(tmp_path / 'test.svg'))
    plt.close()
    mock.assert_called_once()


@pytest.mark.parametrize(
    ('kwargs', 'exception'),
    [
        pytest.param(
            {'arr': np.ones((1000, 3, 3))},
            ValueError,
            id='3_dim_input',
        ),
    ],
)
def test_tsplot_exceptions(kwargs, exception, monkeypatch):
    mock = Mock()
    monkeypatch.setattr(plt, 'show', mock)

    with pytest.raises(exception):
        tsplot(**kwargs)
