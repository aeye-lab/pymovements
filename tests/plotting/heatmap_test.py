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
"""Test heatmap."""
from unittest.mock import Mock

import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import pytest
from matplotlib import figure
from polars import ColumnNotFoundError

from pymovements.gaze import Experiment
from pymovements.gaze.gaze_dataframe import GazeDataFrame
from pymovements.plotting import heatmap


@pytest.fixture(name='experiment_fixture')
def fixture_experiment():
    return Experiment(1024, 768, 38, 30, 60, 'center', 1000)


# Fixture for x and y position columns
@pytest.fixture(
    name='column_names_fixture',
    params=[('x_pos', 'y_pos'), ('x_pix', 'y_pix')],
)
def fixture_column_names(request):
    return request.param


@pytest.fixture(name='args')
def args_fixture(experiment_fixture, column_names_fixture):
    # Init a dataframe with 2 columns and 100 rows
    df = pl.DataFrame(
        {
            column_names_fixture[0]: np.arange(0, 100),
            column_names_fixture[1]: np.arange(0, 100),
        },
    )

    # Init a GazeDataFrame
    gdf = GazeDataFrame(data=df, experiment=experiment_fixture)

    return {'gaze': gdf, 'position_columns': column_names_fixture}


@pytest.mark.parametrize(
    'kwargs',
    [
        pytest.param(
            {'cmap': 'jet'}, id='str_cmap',
        ),
        pytest.param(
            {'cmap': matplotlib.colors.ListedColormap(['red', 'blue', 'green'])}, id='custom_cmap',
        ),
        pytest.param(
            {'gridsize': (10, 10)}, id='default_gridsize',
        ),
        pytest.param(
            {'gridsize': (15, 20)}, id='custom_gridsize',
        ),
        pytest.param(
            {'interpolation': 'gaussian'}, id='default_interpolation',
        ),
        pytest.param(
            {'interpolation': 'bilinear'}, id='custom_interpolation',
        ),
        pytest.param(
            {'origin': 'lower'}, id='default_origin',
        ),
        pytest.param(
            {'origin': 'upper'}, id='custom_origin',
        ),
        pytest.param(
            {
                'title': None,
                'xlabel': None,
                'ylabel': None,
                'cbar_label': None,
            }, id='default_labels',
        ),
        pytest.param(
            {
                'title': 'Custom Title',
                'xlabel': 'Custom X Label',
                'ylabel': 'Custom Y Label',
                'cbar_label': 'Custom Colorbar Label',
            },
            id='custom_labels',
        ),
        pytest.param(
            {'show_cbar': True}, id='show_cbar_true',
        ),
        pytest.param(
            {'show_cbar': False}, id='show_cbar_false',
        ),
    ],
)
def test_heatmap_show(args, kwargs, monkeypatch):
    mock = Mock()
    monkeypatch.setattr(plt, 'show', mock)
    heatmap(**args, **kwargs)
    plt.close()
    mock.assert_called_once()


def test_heatmap_noshow(args, monkeypatch):
    mock = Mock()
    monkeypatch.setattr(plt, 'show', mock)
    heatmap(**args, show=False)
    plt.close()
    mock.assert_not_called()


def test_heatmap_save(args, monkeypatch, tmp_path):
    mock = Mock()
    monkeypatch.setattr(figure.Figure, 'savefig', mock)
    heatmap(**args, show=False, savepath=str(tmp_path / 'test.svg'))
    plt.close()
    mock.assert_called_once()


def test_heatmap_invalid_position_columns(args):
    with pytest.raises(ColumnNotFoundError):
        heatmap(gaze=args['gaze'], position_columns=('x_invalid', 'y_invalid'), show=False)


def test_heatmap_no_experiment_property():
    df = pl.DataFrame(
        {
            'x_pix': np.arange(0, 100),
            'y_pix': np.arange(0, 100),
        },
    )

    gdf = GazeDataFrame(data=df, experiment=None)

    with pytest.raises(ValueError):
        heatmap(gdf, show=False)
