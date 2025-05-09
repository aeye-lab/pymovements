# Copyright (c) 2023-2025 The pymovements Project Authors
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
"""Test for Screen class."""
import pytest

import pymovements as pm


@pytest.mark.parametrize('property_name', ['x_max_dva', 'y_max_dva', 'x_min_dva', 'y_min_dva'])
def test_dva_properties_with_no_distance_cm(property_name):
    screen = pm.Screen(1920, 1080, 30, 20, None, 'upper left')
    with pytest.raises(TypeError):
        getattr(screen, property_name)


@pytest.mark.parametrize('property_name', ['x_max_dva', 'y_max_dva', 'x_min_dva', 'y_min_dva'])
def test_dva_properties_with_distance_cm(property_name):
    screen = pm.Screen(1920, 1080, 30, 20, 60, 'upper left')

    getattr(screen, property_name)


def test_screen_pix2deg_with_no_distance_cm():
    screen = pm.Screen(1920, 1080, 30, 20, None, 'upper left')
    with pytest.raises(TypeError):
        screen.pix2deg([[0, 0]])


def test_screen_pix2deg_with_distance_cm():
    screen = pm.Screen(1920, 1080, 30, 20, 60, 'upper left')
    screen.pix2deg([[0, 0]])


@pytest.mark.parametrize(
    ('missing_attribute', 'exception', 'exception_msg'),
    [
        pytest.param(
            'width_px',
            TypeError,
            "'width_px' must not be None",
            id='width_px',
        ),
        pytest.param(
            'height_px',
            TypeError,
            "'height_px' must not be None",
            id='height_px',
        ),
        pytest.param(
            'width_cm',
            TypeError,
            "'width_cm' must not be None",
            id='width_cm',
        ),
        pytest.param(
            'height_cm',
            TypeError,
            "'height_cm' must not be None",
            id='height_cm',
        ),
        pytest.param(
            'distance_cm',
            TypeError,
            "'distance_cm' must not be None",
            id='distance_cm',
        ),
        pytest.param(
            'origin',
            TypeError,
            "'origin' must not be None",
            id='origin',
        ),
    ],
)
def test_pix2deg_without_attributes(missing_attribute, exception, exception_msg):
    screen = pm.Screen(1920, 1080, 30, 20, 68.0, 'upper left')
    setattr(screen, missing_attribute, None)

    with pytest.raises(exception) as excinfo:
        screen.pix2deg([[0, 0]])

    msg, = excinfo.value.args
    assert msg == exception_msg


def test_screen_init_without_attributes():
    screen = pm.Screen()
    assert isinstance(screen, pm.Screen)


def test_screen_to_dict_exclude_none():
    screen = pm.Screen(1920, None, origin='upper left',)
    new_dict = screen.to_dict()
    assert 'width_px' in new_dict
    assert 'height_px' not in new_dict
    assert 'origin' not in new_dict


def test_screen_bool_all_none():
    assert not bool(pm.Screen())
