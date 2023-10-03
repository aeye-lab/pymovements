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
"""Test for Screen class"""
import pytest

import pymovements as pm


@pytest.mark.parametrize('property_name', ['x_max_dva', 'y_max_dva', 'x_min_dva', 'y_min_dva'])
def test_dva_properties_with_no_distance_cm(property_name):
    screen = pm.Screen(1920, 1080, 30, 20, None, 'lower left')
    with pytest.raises(ValueError):
        getattr(screen, property_name)


@pytest.mark.parametrize('property_name', ['x_max_dva', 'y_max_dva', 'x_min_dva', 'y_min_dva'])
def test_dva_properties_with_distance_cm(property_name):
    screen = pm.Screen(1920, 1080, 30, 20, 60, 'lower left')
    try:
        getattr(screen, property_name)
    except ValueError:
        pytest.fail()


def test_screen_pix2deg_with_no_distance_cm():
    screen = pm.Screen(1920, 1080, 30, 20, None, 'lower left')
    with pytest.raises(ValueError):
        screen.pix2deg([[0, 0]])


def test_screen_pix2deg_with_distance_cm():
    screen = pm.Screen(1920, 1080, 30, 20, 60, 'lower left')
    try:
        screen.pix2deg([[0, 0]])
    except ValueError:
        pytest.fail()
