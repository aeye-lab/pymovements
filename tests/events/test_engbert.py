import pytest

import numpy as np

from pymovements.events.engbert import microsaccades, compute_threshold


@pytest.mark.parametrize(
    'params, expected',
    [
        pytest.param(
            {'velocities': np.random.uniform(size=(10, 2)), 'threshold': (1, 1, 1)},
            {'exception': ValueError},
            id='non_2d_tuple_threshold_raise_value_error',
        ),
        pytest.param(
            {'velocities': np.random.uniform(size=(10, 2)), 'threshold': (0, 100)},
            {'exception': ValueError},
            id='low_variance_yaw_threshold_raise_runtime_error',
        ),
        pytest.param(
            {'velocities': np.random.uniform(size=(10, 2)), 'threshold': (100, 0)},
            {'exception': ValueError},
            id='low_variance_pitch_threshold_raise_runtime_error',
        ),
    ]
)
def test_microsaccades(params, expected):
    if 'exception' in expected:
        with pytest.raises(expected['exception']):
            microsaccades(**params)
        return

    result = microsaccades(**params)
    assert all(result == expected['value'])


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
    ]
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