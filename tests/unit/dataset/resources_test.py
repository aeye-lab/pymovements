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
"""Test dataset resources."""
import pytest

from pymovements import Resources


@pytest.mark.parametrize(
    ('resources1', 'resources2'),
    [
        pytest.param(
            Resources(),
            Resources(),
            id='default',
        ),
        pytest.param(
            Resources(),
            Resources(None),
            id='default_equals_none',
        ),
        pytest.param(
            Resources(tuple()),
            Resources(tuple()),
            id='empty_tuple',
        ),
    ],
)
def test_resources_is_equal(resources1, resources2):
    assert resources1 == resources2


@pytest.mark.parametrize(
    ('resources1', 'resources2'),
    [
        pytest.param(
            Resources(),
            Resources(tuple()),
            id='default_and_empty_tuple',
        ),
    ],
)
def test_resources_is_not_equal(resources1, resources2):
    assert resources1 != resources2


@pytest.mark.parametrize(
    ('init_resources', 'expected_resources'),
    [
        pytest.param(
            None,
            Resources(resources=None),
            id='none',
        ),

        pytest.param(
            {},
            Resources(resources=tuple()),
            id='empty_dict',
        ),

        pytest.param(
            {
                'gaze': [{'filename': 'myfile.txt'}],
            },
            Resources(
                resources=(
                    {'filename': 'myfile.txt', 'content': 'gaze'},
                ),
            ),
            id='single_gaze_resource',
        ),

        pytest.param(
            {
                'gaze': [{'filename': 'myfile1.zip'}, {'filename': 'myfile2.zip'}],
            },
            Resources(
                resources=(
                    {'filename': 'myfile1.zip', 'content': 'gaze'},
                    {'filename': 'myfile2.zip', 'content': 'gaze'},
                ),
            ),
            id='two_gaze_resources',
        ),

        pytest.param(
            {
                'precomputed_events': [{'filename': 'myevents.csv'}],
            },
            Resources(
                resources=(
                    {'filename': 'myevents.csv', 'content': 'precomputed_events'},
                ),
            ),
            id='single_precomputed_events_resource',
        ),

        pytest.param(
            {
                'precomputed_reading_measures': [{'filename': 'reading_measures.csv'}],
            },
            Resources(
                resources=(
                    {
                        'filename': 'reading_measures.csv',
                        'content': 'precomputed_reading_measures',
                    },
                ),
            ),
            id='single_precomputed_events_resource',
        ),

    ],
)
def test_resources_from_dict_expected(init_resources, expected_resources):
    assert Resources.from_dict(init_resources) == expected_resources


@pytest.mark.parametrize(
    ('resources', 'expected_tuple'),
    [
        pytest.param(
            Resources(resources=None),
            None,
            id='none',
        ),

        pytest.param(
            Resources(resources=tuple()),
            tuple(),
            id='empty_dict',
        ),

        pytest.param(
            Resources(
                resources=(
                    {'filename': 'myfile.txt', 'content': 'gaze'},
                ),
            ),
            (
                {'filename': 'myfile.txt', 'content': 'gaze'},
            ),
            id='single_gaze_resource',
        ),

        pytest.param(
            Resources(
                resources=(
                    {'filename': 'myfile1.zip', 'content': 'gaze'},
                    {'filename': 'myfile2.zip', 'content': 'gaze'},
                ),
            ),
            (
                {'filename': 'myfile1.zip', 'content': 'gaze'},
                {'filename': 'myfile2.zip', 'content': 'gaze'},
            ),
            id='two_gaze_resources',
        ),

        pytest.param(
            Resources(
                resources=(
                    {'filename': 'myevents.csv', 'content': 'precomputed_events'},
                ),
            ),
            (
                {'filename': 'myevents.csv', 'content': 'precomputed_events'},
            ),
            id='single_precomputed_events_resource',
        ),

        pytest.param(
            Resources(
                resources=(
                    {
                        'filename': 'reading_measures.csv',
                        'content': 'precomputed_reading_measures',
                    },
                ),
            ),
            (
                {
                    'filename': 'reading_measures.csv',
                    'content': 'precomputed_reading_measures',
                },
            ),
            id='single_precomputed_events_resource',
        ),

    ],
)
def test_resources_to_tuple_expected(resources, expected_tuple):
    assert resources.to_tuple_of_dicts() == expected_tuple


@pytest.mark.parametrize(
    ('resources', 'content_type', 'expected_tuple'),
    [
        pytest.param(
            Resources(),
            None,
            tuple(),
            id='default_none',
        ),

        pytest.param(
            Resources(),
            'gaze',
            tuple(),
            id='default_gaze',
        ),

        pytest.param(
            Resources.from_dicts([{'filename': 'myfile.txt', 'content': 'gaze'}]),
            'gaze',
            ({'filename': 'myfile.txt', 'content': 'gaze'},),
            id='single_gaze_resource_gaze',
        ),

        pytest.param(
            Resources.from_dicts([{'filename': 'myfile.txt', 'content': 'gaze'}]),
            'precomputed_events',
            tuple(),
            id='single_gaze_resource_precomputed_events',
        ),

        pytest.param(
            Resources.from_dicts([{'filename': 'events.csv', 'content': 'precomputed_events'}]),
            'precomputed_events',
            ({'filename': 'events.csv', 'content': 'precomputed_events'},),
            id='single_precomputed_events_resource_precomputed_events',
        ),

        pytest.param(
            Resources.from_dicts([{'filename': 'events.csv', 'content': 'precomputed_events'}]),
            'gaze',
            tuple(),
            id='single_precomputed_events_resource_gaze',
        ),

    ],
)
def test_resources_get_expected(resources, content_type, expected_tuple):
    assert resources.get(content_type) == expected_tuple
