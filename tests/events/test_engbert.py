"""Test all functions in pymovements.events.engbert."""
import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from pymovements.events.engbert import compute_threshold
from pymovements.events.engbert import microsaccades
from pymovements.synthetic import step_function


@pytest.mark.parametrize(
    'kwargs, expected',
    [
        pytest.param(
            {'velocities': np.random.uniform(size=(10, 2)), 'threshold': (1, 1, 1)},
            ValueError,
            id='non_2d_tuple_threshold_raise_value_error',
        ),
        pytest.param(
            {'velocities': np.random.uniform(size=(10, 2)), 'threshold': (0, 100)},
            ValueError,
            id='low_variance_yaw_threshold_raise_runtime_error',
        ),
        pytest.param(
            {'velocities': np.random.uniform(size=(10, 2)), 'threshold': (100, 0)},
            ValueError,
            id='low_variance_pitch_threshold_raise_runtime_error',
        ),
    ],
)
def test_microsaccades_raises_error(kwargs, expected):
    with pytest.raises(expected):
        microsaccades(**kwargs)


@pytest.mark.parametrize(
    'kwargs, expected',
    [
        pytest.param(
            {
                'velocities': step_function(
                    length=100,
                    steps=[40, 50],
                    values=[(9, 9), (0, 0)],
                    start_value=(0, 0),
                ),
                'threshold': 1e-5,
            },
            pl.DataFrame(
                {
                    'type': 'saccade',
                    'onset': [40],
                    'offset': [49],
                },
            ),
            id='two_steps_one_saccade',
        ),
        pytest.param(
            {
                'velocities': step_function(
                    length=100,
                    steps=[20, 30, 70, 80],
                    values=[(9, 9), (0, 0), (9, 9), (0, 0)],
                    start_value=(0, 0),
                ),
                'threshold': 1e-5,
            },
            pl.DataFrame(
                {
                    'type': 'saccade',
                    'onset': [20, 70],
                    'offset': [29, 79],
                },
            ),
            id='four_steps_two_saccades',
        ),
    ],
)
def test_microsaccades_detects_saccades(kwargs, expected):
    events = microsaccades(**kwargs)

    assert_frame_equal(events, expected)


@pytest.mark.parametrize(
    'params, expected',
    [
        pytest.param(
            {'method': 'invalid'},
            {'exception': ValueError},
            id='invalid_method_raises_value_error',
        ),
        pytest.param(
            {'method': 'std'},
            {'value': (1.16619038, 1.16619038)},
            id='std',
        ),
        pytest.param(
            {'method': 'mad'},
            {'value': (1, 1)},
            id='mad',
        ),
        pytest.param(
            {'method': 'engbert2003'},
            {'value': np.array((1., 1.))},
            id='engbert2003',
        ),
        pytest.param(
            {'method': 'engbert2015'},
            {'value': np.array((1., 1.))},
            id='engbert2015',
        ),
    ],
)
def test_compute_threshold(params, expected):
    v = np.empty((101, 2))
    v[:, 0] = np.linspace(-2, 2, 101)
    v[:, 1] = np.linspace(-2, 2, 101)

    if 'exception' in expected:
        with pytest.raises(expected['exception']):
            compute_threshold(arr=v, **params)
        return

    result = compute_threshold(arr=v, **params)
    assert np.allclose(result, expected['value'])
