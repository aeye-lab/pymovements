"""
Test all functions in pymovements.transforms.
"""
# pylint: disable=missing-function-docstring
import numpy as np
import pytest

from pymovements.transforms import pix2deg
from pymovements.transforms import pos2vel


def test_pix2deg_none_coords_raise_type_error():
    with pytest.raises(TypeError):
        pix2deg(None, 1, 1, 1)


def test_pix2deg_none_screen_px_raise_type_error():
    with pytest.raises(TypeError):
        pix2deg(0, None, 1, 1)


def test_pix2deg_none_screen_cm_raise_type_error():
    with pytest.raises(TypeError):
        pix2deg(0, 1, None, 1)


def test_pix2deg_none_distance_cm_raise_type_error():
    with pytest.raises(TypeError):
        pix2deg(0, 1, 1, None)


def test_pix2deg_zero_screen_px_raise_value_error():
    with pytest.raises(ValueError):
        pix2deg(0, 0, 1, 1)


def test_pix2deg_zero_screen_cm_raise_value_error():
    with pytest.raises(ValueError):
        pix2deg(0, 1, 0.0, 1)


def test_pix2deg_zero_distance_cm_raise_value_error():
    with pytest.raises(ValueError):
        pix2deg(0, 1, 1, 0.0)


def test_pix2deg_rank_3_tensor_raises_value_error():
    with pytest.raises(ValueError):
        coords = np.zeros((10, 2, 2))
        pix2deg(coords, [100, 100], [100, 100], 100, False)


def test_pix2deg_zero_coord_without_center_origin_equals_zero():
    assert pix2deg(0, 100, 100, 100, False) == 0


def test_pix2deg_center_coord_with_center_origin_equals_zero():
    screen_px = 100
    center_px = (screen_px - 1) / 2
    assert pix2deg(center_px, screen_px, 10, 10, True) == 0


def test_pix2deg_isosceles_triangle_without_center_origin_equals_45():
    screen_px = 100
    screen_cm = 100
    distance_cm = screen_cm / 2
    coord = screen_px / 2

    assert pix2deg(coord, screen_px, screen_cm, distance_cm, False) == 45


def test_pix2deg_isosceles_triangle_with_center_origin_equals_45():
    screen_px = 100
    screen_cm = 100
    distance_cm = screen_cm / 2
    coord = screen_px - 0.5

    assert pix2deg(coord, screen_px, screen_cm, distance_cm, True) == 45


def test_pix2deg_isosceles_triangle_left_with_center_origin_equals_minus45():
    screen_px = 100
    screen_cm = 100
    distance_cm = screen_cm / 2
    coord = -0.5

    assert pix2deg(coord, screen_px, screen_cm, distance_cm, True) == -45


def test_pix2deg_ankathet_half_without_center_origin_equals_26565():
    screen_px = 100
    screen_cm = 100
    distance_cm = screen_cm
    coord = screen_px / 2

    value = pix2deg(coord, screen_px, screen_cm, distance_cm, False)
    assert value == pytest.approx(26.565, abs=1e-4)


def test_pix2deg_ankathet_half_with_center_origin_equals_26565():
    screen_px = 100
    screen_cm = 100
    distance_cm = screen_cm
    coord = screen_px - 0.5

    value = pix2deg(coord, screen_px, screen_cm, distance_cm, True)
    assert value == pytest.approx(26.565, abs=1e-4)


def test_pix2deg_ankathet_sqrt_3_without_center_origin_equals_60():
    screen_px = 100
    screen_cm = 100
    distance_cm = screen_cm / 2 / np.sqrt(3)
    coord = screen_px / 2

    assert (
        pix2deg(coord, screen_px, screen_cm, distance_cm, False)
        == pytest.approx(60)
    )


def test_pix2deg_ankathet_sqrt_3_with_center_origin_equals_60():
    screen_px = 100
    screen_cm = 100
    distance_cm = screen_cm / 2 / np.sqrt(3)
    coord = screen_px - 0.5

    assert (
        pix2deg(coord, screen_px, screen_cm, distance_cm, True)
        == pytest.approx(60)
    )


def test_pix2deg_opposite_sqrt_3_without_center_origin_equals_30():
    screen_px = 100
    screen_cm = 100
    distance_cm = screen_cm / 2 * np.sqrt(3)
    coord = screen_px / 2

    actual_value = pix2deg(coord, screen_px, screen_cm, distance_cm, False)
    assert actual_value == pytest.approx(30)


def test_pix2deg_opposite_sqrt_3_with_center_origin_equals_30():
    screen_px = 100
    screen_cm = 100
    distance_cm = screen_cm / 2 * np.sqrt(3)
    coord = screen_px - 0.5

    actual_value = pix2deg(coord, screen_px, screen_cm, distance_cm, True)
    assert actual_value == pytest.approx(30)


def test_pix2deg_list_of_zero_coords_1d():
    n_coords = 10
    coords = [0] * n_coords
    control = np.array([0.0] * n_coords)
    actual = pix2deg(coords, 100, 100, 100, False)
    assert (control == actual).all()


def test_pix2deg_nparray_of_zero_coords_1d():
    n_coords = 10
    coords = np.array([0] * n_coords)
    control = np.array([0.0] * n_coords)
    actual = pix2deg(coords, 100, 100, 100, False)
    assert (control == actual).all()


def test_pix2deg_list_of_zero_coords_2d():
    n_coords = 10
    coords = [[0, 0]] * n_coords
    control = np.array([[0.0, 0.0]] * n_coords)
    actual = pix2deg(coords, [100, 100], [100, 100], 100, False)
    assert (control == actual).all()


def test_pix2deg_list_of_zero_coords_2d_screen_px_1d_raise_value_error():
    n_coords = 10
    coords = [[0, 0]] * n_coords
    with pytest.raises(ValueError):
        pix2deg(coords, 100, [100, 100], 100, False)


def test_pix2deg_list_of_zero_coords_2d_screen_cm_1d_raise_value_error():
    n_coords = 10
    coords = [[0, 0]] * n_coords
    with pytest.raises(ValueError):
        pix2deg(coords, [100, 100], 100, 100, False)


def test_pix2deg_nparray_of_isosceles_triangle_without_center_origin_equals_45():
    screen_px = [100, 100]
    screen_cm = [100, 100]
    distance_cm = screen_cm[0] / 2
    coord = screen_px[0] / 2 * np.ones((10, 2))

    control = 45
    actual = pix2deg(coord, screen_px, screen_cm, distance_cm, False)
    assert (actual == control).all()


def test_pix2deg_list_1d_screen_px_2d_raise_value_error():
    with pytest.raises(ValueError):
        pix2deg([0] * 10, [100, 100], 100, 100)


def test_pix2deg_list_1d_screen_cm_2d_raise_value_error():
    with pytest.raises(ValueError):
        pix2deg([0] * 10, 100, [100, 100], 100)


def test_pix2deg_list_3d_raise_value_error():
    with pytest.raises(ValueError):
        pix2deg(np.zeros((10, 3)), [1, 1, 1], [1, 1, 1], 1)


def test_pos2vel_sampling_rate_zero_raise_value_error():
    with pytest.raises(ValueError):
        pos2vel([0]*10, sampling_rate=0)


def test_pos2vel_sampling_rate_less_zero_raise_value_error():
    with pytest.raises(ValueError):
        pos2vel([0]*10, sampling_rate=-1)


def test_pos2vel_list_length_below_six_method_smooth_raise_value_error():
    with pytest.raises(ValueError):
        pos2vel([0]*5, method='smooth')


def test_pos2vel_list_length_below_six_method_neighbors_raise_value_error():
    with pytest.raises(ValueError):
        pos2vel([0]*2, method='neighbors')


def test_pos2vel_list_length_below_six_method_preceding_raise_value_error():
    with pytest.raises(ValueError):
        pos2vel([0], method='preceding')


def test_pos2vel_invalid_method_raises_value_error():
    with pytest.raises(ValueError):
        pos2vel([0]*10, method='invalid')


@pytest.mark.parametrize('method', ['smooth', 'neighbors', 'preceding'])
def test_pos2vel_constant_input_returns_zero_velocity(method):
    N = 10
    x = np.repeat(0, N)
    actual_value = pos2vel(x, sampling_rate=1, method=method)
    control_value = np.zeros(x.shape)

    assert (actual_value == control_value).all()


@pytest.mark.parametrize('method', ['smooth', 'neighbors', 'preceding'])
def test_pos2vel_linear_input_returns_constant_velocity(method):
    N = 100
    x = np.linspace(0, N - 1, N)
    actual_value = pos2vel(x, sampling_rate=1, method=method)
    control_value = np.ones(x.shape)

    lpad, rpad = 1, -1
    assert (actual_value[lpad:rpad] == control_value[lpad:rpad]).all()


def test_pos2vel_stepped_input_returns_alternating_velocity_method_preceding():
    N = 100
    x = np.linspace(0, N - 2, N // 2)
    x = np.repeat(x, 2)

    actual_value = pos2vel(x, sampling_rate=1, method='preceding')
    control_value = np.array([2.0, 0.0] * (N//2))

    lpad, rpad = 1, -1
    assert (actual_value[lpad:rpad] == control_value[lpad:rpad]).all()


def test_pos2vel_stepped_input_returns_constant_velocity_method_neighbors():
    N = 100
    x = np.linspace(0, N - 2, N // 2)
    x = np.repeat(x, 2)

    actual_value = pos2vel(x, sampling_rate=1, method='neighbors')
    control_value = np.ones(x.shape)

    lpad, rpad = 1, -1
    assert (actual_value[lpad:rpad] == control_value[lpad:rpad]).all()


def test_pos2vel_stepped_input_returns_constant_velocity_method_smooth():
    N = 100
    x = np.linspace(0, N - 2, N // 2)
    x = np.repeat(x, 2)

    actual_value = pos2vel(x, sampling_rate=1, method='smooth')
    control_value = np.ones(x.shape)

    lpad, rpad = 1, -1
    assert (actual_value[lpad:rpad] == control_value[lpad:rpad]).all()
