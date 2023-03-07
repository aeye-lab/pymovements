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
from unittest.mock import Mock

import matplotlib.colors
import matplotlib.figure as figure
import matplotlib.pyplot as plt
import numpy as np
import pytest

from pymovements.plotting import traceplot


@pytest.fixture(name='args')
def args_fixture():
    x = np.arange(-100, 100)
    y = np.arange(-100, 100)

    return x, y


@pytest.mark.parametrize(
    'kwargs',
    [
        pytest.param({}, id='no_kwargs'),
        pytest.param({'cval': np.arange(-100, 100)}, id='cval_array'),
        pytest.param(
            {'cval': np.arange(-100, 100), 'cmap_norm': 'twoslope'},
            id='cmap_norm_twoslope',
        ),
        pytest.param(
            {'cval': np.arange(0, 200), 'cmap_norm': 'nonorm'},
            id='cmap_norm_nonorm',
        ),
        pytest.param(
            {'cval': np.arange(0, 200)},
            id='cmap_norm_nonorm_implicit',
        ),
        pytest.param(
            {'cval': np.arange(-100, 100), 'cmap_norm': 'normalize'},
            id='cmap_norm_normalize',
        ),
        pytest.param(
            {'cval': np.arange(0, 200), 'cmap_norm': 'linear'},
            id='cmap_norm_linear',
        ),
        pytest.param(
            {'cmap': matplotlib.colors.LinearSegmentedColormap(name='test', segmentdata={})},
            id='cmap_class',
        ),
        pytest.param(
            {'cmap_segmentdata': {}},
            id='cmap_class',
        ),
        pytest.param({'padding': 0.1}, id='padding'),
        pytest.param({'cval': np.arange(0, 200), 'show_cbar': True}, id='show_cbar_true'),
        pytest.param({'cval': np.arange(0, 200), 'show_cbar': False}, id='show_cbar_false'),
    ],
)
def test_traceplot_show(args, kwargs, monkeypatch):
    mock = Mock()
    monkeypatch.setattr(plt, 'show', mock)
    traceplot(*args, **kwargs)
    mock.assert_called_once()


def test_traceplot_noshow(args, monkeypatch):
    mock = Mock()
    monkeypatch.setattr(plt, 'show', mock)
    traceplot(*args, show=False)
    mock.assert_not_called()


def test_traceplot_save(args, monkeypatch, tmp_path):
    mock = Mock()
    monkeypatch.setattr(figure.Figure, 'savefig', mock)
    traceplot(*args, show=False, savepath=str(tmp_path / 'test.svg'))
    mock.assert_called_once()


@pytest.mark.parametrize(
    ('kwargs', 'exception'),
    [
        pytest.param(
            {'x': np.arange(100), 'y': np.arange(11)},
            ValueError,
            id='different_lengths',
        ),
    ],
)
def test_traceplot_exceptions(kwargs, exception, monkeypatch):
    mock = Mock()
    monkeypatch.setattr(plt, 'show', mock)

    with pytest.raises(exception):
        traceplot(**kwargs)
