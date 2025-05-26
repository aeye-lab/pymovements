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

from pymovements import Resource
from pymovements import Resources


@pytest.mark.parametrize(
    'kwargs',
    [
        pytest.param(
            {
                'content': 'gaze',
                'filename': 'test.csv',
            },
            id='gaze_content_filename',
        ),

        pytest.param(
            {
                'content': 'gaze',
                'filename': 'test.csv',
                'url': 'https://example.com',
            },
            id='gaze_content_filename_url',
        ),

        pytest.param(
            {
                'content': 'gaze',
                'filename': 'test.csv',
                'url': 'https://example.com',
                'md5': 'abcdefgh',
            },
            id='gaze_content_filename_url_md5',
        ),
    ],
)
def test_resource_is_equal(kwargs):
    assert Resource(**kwargs) == Resource(**kwargs)


@pytest.mark.parametrize(
    ('resource1', 'resource2'),
    [
        pytest.param(
            Resource(content='gaze', filename='test1.csv'),
            Resource(content='gaze', filename='test2.csv'),
            id='different_filename',
        ),

        pytest.param(
            Resource(content='gaze', filename='test.csv'),
            Resource(content='precomputed_events', filename='test.csv'),
            id='different_content',
        ),

        pytest.param(
            Resource(content='gaze', filename='test.csv', url='https://example.com'),
            Resource(content='gaze', filename='test.csv', url='https://examples.com'),
            id='different_url',
        ),

        pytest.param(
            Resource(
                content='gaze',
                filename='test.csv',
                url='https://example.com',
                md5='abcdefgh',
            ),
            Resource(
                content='gaze',
                filename='test.csv',
                url='https://example.com',
                md5='ijklmnop',
            ),
            id='different_md5',
        ),
    ],
)
def test_resource_is_not_equal(resource1, resource2):
    assert resource1 != resource2


@pytest.mark.parametrize(
    ('resource_dict', 'expected_resource'),
    [
        pytest.param(
            {
                'content': 'gaze',
            },
            Resource(
                content='gaze',
                filename=None,
                url=None,
                md5=None,
            ),
            id='content',
        ),

        pytest.param(
            {
                'content': 'gaze',
                'filename': 'test.csv',
            },
            Resource(
                content='gaze',
                filename='test.csv',
                url=None,
                md5=None,
            ),
            id='content_filename',
        ),

        pytest.param(
            {
                'content': 'gaze',
                'filename': 'test.csv',
                'url': 'https://example.com',
            },
            Resource(
                content='gaze',
                filename='test.csv',
                url='https://example.com',
                md5=None,
            ),
            id='content_filename_url',
        ),

        pytest.param(
            {
                'content': 'gaze',
                'filename': 'test.csv',
                'url': 'https://example.com',
                'md5': 'abcdefgh',
            },
            Resource(
                content='gaze',
                filename='test.csv',
                url='https://example.com',
                md5='abcdefgh',
            ),
            id='content_filename_url_md5',
        ),

        pytest.param(
            {
                'content': 'gaze',
                'filename': 'test.csv',
                'resource': 'https://example.com',
            },
            Resource(
                content='gaze',
                filename='test.csv',
                url='https://example.com',
                md5=None,
            ),
            id='deprecated_resource_key',
        ),

        pytest.param(
            {
                'content': 'gaze',
                'filename_pattern': 'test.csv',
            },
            Resource(
                content='gaze',
                filename=None,
                url=None,
                md5=None,
                filename_pattern='test.csv',
            ),
            id='filename_pattern',
        ),

        pytest.param(
            {
                'content': 'gaze',
                'filename_pattern': '{subject_id:d}.csv',
                'filename_pattern_schema_overrides': {'subject_id': int},
            },
            Resource(
                content='gaze',
                filename=None,
                url=None,
                md5=None,
                filename_pattern='{subject_id:d}.csv',
                filename_pattern_schema_overrides={'subject_id': int},
            ),
            id='filename_pattern_schema_overrides',
        ),

    ],
)
def test_resource_from_dict_expected(resource_dict, expected_resource):
    assert Resource.from_dict(resource_dict) == expected_resource


@pytest.mark.parametrize(
    ('init_resources', 'expected_resources'),
    [
        pytest.param(
            None,
            Resources(),
            id='none',
        ),

        pytest.param(
            {
                'gaze': None,
            },
            Resources(),
            id='none_gaze_list',
        ),

        pytest.param(
            {
                'gaze': [],
            },
            Resources(),
            id='empty_gaze_list',
        ),

        pytest.param(
            {
                'gaze': [{'filename': 'myfile.txt'}],
            },
            Resources(
                [
                    Resource(filename='myfile.txt', content='gaze'),
                ],
            ),
            id='single_gaze_resource',
        ),

        pytest.param(
            {
                'gaze': [{'filename': 'myfile1.zip'}, {'filename': 'myfile2.zip'}],
            },
            Resources(
                [
                    Resource(filename='myfile1.zip', content='gaze'),
                    Resource(filename='myfile2.zip', content='gaze'),
                ],
            ),
            id='two_gaze_resources',
        ),

        pytest.param(
            {
                'precomputed_events': [{'filename': 'myevents.csv'}],
            },
            Resources(
                [
                    Resource(filename='myevents.csv', content='precomputed_events'),
                ],
            ),
            id='single_precomputed_events_resource',
        ),

        pytest.param(
            {
                'precomputed_reading_measures': [{'filename': 'reading_measures.csv'}],
            },
            Resources(
                [
                    Resource(
                        filename='reading_measures.csv',
                        content='precomputed_reading_measures',
                    ),
                ],
            ),
            id='single_precomputed_events_resource',
        ),

    ],
)
def test_resources_from_dict_expected(init_resources, expected_resources):
    assert Resources.from_dict(init_resources) == expected_resources


@pytest.mark.parametrize(
    ('resources', 'expected_dicts'),
    [
        pytest.param(
            Resources(),
            [],
            id='default',
        ),

        pytest.param(
            Resources(
                [
                    Resource(filename='myfile.txt', content='gaze'),
                ],
            ),
            [
                {'filename': 'myfile.txt', 'content': 'gaze'},
            ],
            id='single_gaze_resource',
        ),

        pytest.param(
            Resources(
                [
                    Resource(filename='myfile1.zip', content='gaze'),
                    Resource(filename='myfile2.zip', content='gaze'),
                ],
            ),
            [
                {'filename': 'myfile1.zip', 'content': 'gaze'},
                {'filename': 'myfile2.zip', 'content': 'gaze'},
            ],
            id='two_gaze_resources',
        ),

        pytest.param(
            Resources(
                [
                    Resource(filename='myevents.csv', content='precomputed_events'),
                ],
            ),
            [
                {'filename': 'myevents.csv', 'content': 'precomputed_events'},
            ],
            id='single_precomputed_events_resource',
        ),

        pytest.param(
            Resources(
                [
                    Resource(
                        filename='reading_measures.csv',
                        content='precomputed_reading_measures',
                    ),
                ],
            ),
            [
                {
                    'filename': 'reading_measures.csv',
                    'content': 'precomputed_reading_measures',
                },
            ],
            id='single_precomputed_events_resource',
        ),

    ],
)
def test_resources_to_dicts_expected(resources, expected_dicts):
    assert resources.to_dicts() == expected_dicts


@pytest.mark.parametrize(
    ('resources', 'content_type', 'expected_resources'),
    [
        pytest.param(
            Resources(),
            None,
            [],
            id='default_filter_none',
        ),

        pytest.param(
            Resources(),
            'gaze',
            [],
            id='default_filter_gaze',
        ),

        pytest.param(
            Resources.from_dicts([{'filename': 'myfile.txt', 'content': 'gaze'}]),
            'gaze',
            [
                Resource(filename='myfile.txt', content='gaze'),
            ],
            id='single_gaze_filter_gaze',
        ),

        pytest.param(
            Resources.from_dicts([{'filename': 'myfile.txt', 'content': 'gaze'}]),
            'precomputed_events',
            [],
            id='single_gaze_filter_precomputed_events',
        ),

        pytest.param(
            Resources.from_dicts([{'filename': 'events.csv', 'content': 'precomputed_events'}]),
            'precomputed_events',
            [
                Resource(filename='events.csv', content='precomputed_events'),
            ],
            id='single_precomputed_events_filter_precomputed_events',
        ),

        pytest.param(
            Resources.from_dicts([{'filename': 'events.csv', 'content': 'precomputed_events'}]),
            'gaze',
            [],
            id='single_precomputed_events_filter_gaze',
        ),

        pytest.param(
            Resources.from_dicts(
                [
                    {'filename': 'myfile.txt', 'content': 'gaze'},
                    {'filename': 'events.csv', 'content': 'precomputed_events'},
                ],
            ),
            None,
            [
                Resource(filename='myfile.txt', content='gaze'),
                Resource(filename='events.csv', content='precomputed_events'),
            ],
            id='gaze_and_precomputed_events_filter_none',
        ),

        pytest.param(
            Resources.from_dicts(
                [
                    {'filename': 'myfile.txt', 'content': 'gaze'},
                    {'filename': 'events.csv', 'content': 'precomputed_events'},
                ],
            ),
            'gaze',
            [
                Resource(filename='myfile.txt', content='gaze'),
            ],
            id='gaze_and_precomputed_events_filter_gaze',
        ),

        pytest.param(
            Resources.from_dicts(
                [
                    {'filename': 'myfile.txt', 'content': 'gaze'},
                    {'filename': 'events.csv', 'content': 'precomputed_events'},
                ],
            ),
            'precomputed_events',
            [
                Resource(filename='events.csv', content='precomputed_events'),
            ],
            id='gaze_and_precomputed_events_filter_precomputed_events',
        ),

    ],
)
def test_resources_filter_expected(resources, content_type, expected_resources):
    assert resources.filter(content_type) == expected_resources


@pytest.mark.parametrize(
    ('resources', 'expected_has_content'),
    [
        pytest.param(
            Resources(),
            {
                'gaze': False,
                'precomputed_events': False,
                'precomputed_reading_measures': False,
            },
            id='default',
        ),

        pytest.param(
            Resources([]),
            {
                'gaze': False,
                'precomputed_events': False,
                'precomputed_reading_measures': False,
            },
            id='empty_list',
        ),

        pytest.param(
            Resources([Resource(content='gaze')]),
            {
                'gaze': True,
                'precomputed_events': False,
                'precomputed_reading_measures': False,
            },
            id='gaze_resources',
        ),

        pytest.param(
            Resources([Resource(content='precomputed_events')]),
            {
                'gaze': False,
                'precomputed_events': True,
                'precomputed_reading_measures': False,
            },
            id='precomputed_event_resources',
        ),

        pytest.param(
            Resources([Resource(content='precomputed_reading_measures')]),
            {
                'gaze': False,
                'precomputed_events': False,
                'precomputed_reading_measures': True,
            },
            id='precomputed_reading_measures_resources',
        ),

        pytest.param(
            Resources(
                [
                    Resource(content='gaze'),
                    Resource(content='precomputed_events'),
                    Resource(content='precomputed_reading_measures'),
                ],
            ),
            {
                'gaze': True,
                'precomputed_events': True,
                'precomputed_reading_measures': True,
            },
            id='all_resources',
        ),

        pytest.param(
            Resources([Resource(content='foo')]),
            {
                'foo': True,
                'gaze': False,
                'precomputed_events': False,
                'precomputed_reading_measures': False,
            },
            id='custom_content',
        ),
    ],
)
def test_resources_has_content_expected(resources, expected_has_content):
    for key, value in expected_has_content.items():
        assert resources.has_content(key) == value, (resources, value)
