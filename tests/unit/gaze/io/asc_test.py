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
"""Test read from eyelink asc files."""
import polars as pl
import pytest

import pymovements as pm


@pytest.mark.parametrize(
    ('kwargs', 'shape', 'schema'),
    [
        pytest.param(
            {
                'file': 'tests/files/eyelink_monocular_example.asc',
                'patterns': 'eyelink',
            },
            (16, 3),
            {
                'time': pl.Int64,
                'pupil': pl.Float64,
                'pixel': pl.List(pl.Float64),
            },
            id='eyelink_asc_mono_pattern_eyelink',
        ),
        pytest.param(
            {
                'file': 'tests/files/eyelink_monocular_example.asc',
                'patterns': pm.datasets.ToyDatasetEyeLink().custom_read_kwargs['patterns'],
                'schema': pm.datasets.ToyDatasetEyeLink().custom_read_kwargs['schema'],
            },
            (16, 7),
            {
                'time': pl.Int64,
                'pupil': pl.Float64,
                'pixel': pl.List(pl.Float64),
                'trial_id': pl.Int64,
                'point_id': pl.Int64,
                'screen_id': pl.Int64,
                'task': pl.Utf8,
            },
            id='eyelink_asc_mono_pattern_list',
        ),
    ],
)
def test_from_asc_has_shape_and_schema(kwargs, shape, schema):
    gaze_dataframe = pm.gaze.from_asc(**kwargs)

    assert gaze_dataframe.frame.shape == shape
    assert gaze_dataframe.frame.schema == schema


@pytest.mark.parametrize(
    ('kwargs', 'exception', 'message'),
    [
        pytest.param(
            {
                'file': 'tests/files/eyelink_monocular_example.asc',
                'patterns': 'foobar',
            },
            ValueError,
            "unknown pattern key 'foobar'. Supported keys are: eyelink",
            id='unknown_pattern',
        ),
    ],
)
def test_from_asc_raises_exception(kwargs, exception, message):
    with pytest.raises(exception) as excinfo:
        pm.gaze.from_asc(**kwargs)

    msg, = excinfo.value.args
    assert msg == message
