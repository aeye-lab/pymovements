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
"""Test read from eyelink asc files."""
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from pymovements import DatasetDefinition
from pymovements import Experiment
from pymovements import EyeTracker
from pymovements import Screen
from pymovements.datasets import ToyDatasetEyeLink
from pymovements.gaze import from_asc


@pytest.mark.parametrize(
    ('kwargs', 'expected_frame'),
    [
        pytest.param(
            {
                'file': 'tests/files/eyelink_monocular_example.asc',
                'patterns': 'eyelink',
            },
            pl.from_dict(
                data={
                    'time': [
                        2154556, 2154557, 2154560, 2154564, 2154596, 2154598, 2154599, 2154695,
                        2154696, 2339227, 2339245, 2339246, 2339271, 2339272, 2339290, 2339291,
                    ],
                    'pupil': [
                        778.0, 778.0, 777.0, 778.0, 784.0, 784.0, 784.0, 798.0,
                        799.0, 619.0, 621.0, 622.0, 617.0, 617.0, 618.0, 618.0,
                    ],
                    'pixel': [
                        [138.1, 132.8], [138.2, 132.7], [137.9, 131.6], [138.1, 131.0],
                        [139.6, 132.1], [139.5, 131.9], [139.5, 131.8], [147.2, 134.4],
                        [147.3, 134.1], [673.2, 523.8], [629.0, 531.4], [629.9, 531.9],
                        [639.4, 531.9], [639.0, 531.9], [637.6, 531.4], [637.3, 531.2],
                    ],
                },
                schema={
                    'time': pl.Int64,
                    'pupil': pl.Float64,
                    'pixel': pl.List(pl.Float64),
                },
            ),
            id='eyelink_asc_mono_pattern_eyelink',
        ),

        pytest.param(
            {
                'file': 'tests/files/eyelink_monocular_example.asc',
                'definition': ToyDatasetEyeLink(),
            },
            pl.DataFrame(
                data={
                    'time': [
                        2154556, 2154557, 2154560, 2154564, 2154596, 2154598, 2154599, 2154695,
                        2154696, 2339227, 2339245, 2339246, 2339271, 2339272, 2339290, 2339291,
                    ],
                    'pupil': [
                        778.0, 778.0, 777.0, 778.0, 784.0, 784.0, 784.0, 798.0,
                        799.0, 619.0, 621.0, 622.0, 617.0, 617.0, 618.0, 618.0,
                    ],
                    'pixel': [
                        [138.1, 132.8], [138.2, 132.7], [137.9, 131.6], [138.1, 131.0],
                        [139.6, 132.1], [139.5, 131.9], [139.5, 131.8], [147.2, 134.4],
                        [147.3, 134.1], [673.2, 523.8], [629.0, 531.4], [629.9, 531.9],
                        [639.4, 531.9], [639.0, 531.9], [637.6, 531.4], [637.3, 531.2],
                    ],
                    'trial_id': [0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 3, 3, 4, 4, None],
                    'point_id': 3 * [None] + [0, 1, 2, 3] + [None, 0] + [0, 0, 1, 2] + [0, 1, None],
                    'screen_id': [None, 0, 1] + 13 * [None],
                    'task': [None] + 2 * ['reading'] + 12 * ['judo'] + [None],
                },
                schema={
                    'time': pl.Int64,
                    'pupil': pl.Float64,
                    'task': pl.Utf8,
                    'screen_id': pl.Int64,
                    'point_id': pl.Int64,
                    'trial_id': pl.Int64,
                    'pixel': pl.List(pl.Float64),
                },
            ),
            id='eyelink_asc_mono_pattern_list',
        ),

        pytest.param(
            {
                'file': 'tests/files/eyelink_monocular_example.asc',
                'definition': ToyDatasetEyeLink(
                    trial_columns=None,
                    custom_read_kwargs={'gaze': {'column_schema_overrides': {'pupil': pl.Float32}}},
                ),
            },
            pl.DataFrame(
                data={
                    'time': [
                        2154556, 2154557, 2154560, 2154564, 2154596, 2154598, 2154599, 2154695,
                        2154696, 2339227, 2339245, 2339246, 2339271, 2339272, 2339290, 2339291,
                    ],
                    'pupil': [
                        778.0, 778.0, 777.0, 778.0, 784.0, 784.0, 784.0, 798.0,
                        799.0, 619.0, 621.0, 622.0, 617.0, 617.0, 618.0, 618.0,
                    ],
                    'pixel': [
                        [138.1, 132.8], [138.2, 132.7], [137.9, 131.6], [138.1, 131.0],
                        [139.6, 132.1], [139.5, 131.9], [139.5, 131.8], [147.2, 134.4],
                        [147.3, 134.1], [673.2, 523.8], [629.0, 531.4], [629.9, 531.9],
                        [639.4, 531.9], [639.0, 531.9], [637.6, 531.4], [637.3, 531.2],
                    ],
                },
                schema={
                    'time': pl.Int64,
                    'pupil': pl.Float32,
                    'pixel': pl.List(pl.Float64),
                },
            ),
            id='eyelink_asc_mono_schema_overrides',
        ),

        pytest.param(
            {
                'file': 'tests/files/eyelink_monocular_example.asc',
                'definition': ToyDatasetEyeLink(
                    trial_columns=None,
                    custom_read_kwargs={'gaze': {'column_schema_overrides': {'pupil': pl.Float32}}},
                ),
                'column_schema_overrides': {'pupil': pl.Decimal},
            },
            pl.DataFrame(
                data={
                    'time': [
                        2154556, 2154557, 2154560, 2154564, 2154596, 2154598, 2154599, 2154695,
                        2154696, 2339227, 2339245, 2339246, 2339271, 2339272, 2339290, 2339291,
                    ],
                    'pupil': [
                        778.0, 778.0, 777.0, 778.0, 784.0, 784.0, 784.0, 798.0,
                        799.0, 619.0, 621.0, 622.0, 617.0, 617.0, 618.0, 618.0,
                    ],
                    'pixel': [
                        [138.1, 132.8], [138.2, 132.7], [137.9, 131.6], [138.1, 131.0],
                        [139.6, 132.1], [139.5, 131.9], [139.5, 131.8], [147.2, 134.4],
                        [147.3, 134.1], [673.2, 523.8], [629.0, 531.4], [629.9, 531.9],
                        [639.4, 531.9], [639.0, 531.9], [637.6, 531.4], [637.3, 531.2],
                    ],
                },
                schema={
                    'time': pl.Int64,
                    'pupil': pl.Decimal,
                    'pixel': pl.List(pl.Float64),
                },
            ),
            id='eyelink_asc_mono_schema_overrides_overrides_definition',
        ),

        pytest.param(
            {
                'file': 'tests/files/eyelink_monocular_2khz_example.asc',
                'patterns': 'eyelink',
            },
            pl.from_dict(
                data={
                    'time': [
                        2154556.5, 2154557.0, 2154560.5, 2154564.0, 2154596.0, 2154598.5, 2154599.0,
                        2154695.0, 2154696.0, 2339227.0, 2339245.0, 2339246.0, 2339271.5, 2339272.0,
                        2339290.0, 2339291.0,
                    ],
                    'pupil': [
                        778.0, 778.0, 777.0, 778.0, 784.0, 784.0, 784.0, 798.0,
                        799.0, 619.0, 621.0, 622.0, 617.0, 617.0, 618.0, 618.0,
                    ],
                    'pixel': [
                        [138.1, 132.8], [138.2, 132.7], [137.9, 131.6], [138.1, 131.0],
                        [139.6, 132.1], [139.5, 131.9], [139.5, 131.8], [147.2, 134.4],
                        [147.3, 134.1], [673.2, 523.8], [629.0, 531.4], [629.9, 531.9],
                        [639.4, 531.9], [639.0, 531.9], [637.6, 531.4], [637.3, 531.2],
                    ],
                },
                schema={
                    'time': pl.Float64,
                    'pupil': pl.Float64,
                    'pixel': pl.List(pl.Float64),
                },
            ),
            id='eyelink_asc_mono_2khz_pattern_eyelink',
        ),
    ],
)
def test_from_asc_has_frame_equal(kwargs, expected_frame):
    gaze = from_asc(**kwargs)

    assert_frame_equal(gaze.samples, expected_frame, check_column_order=False)


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
                'definition': ToyDatasetEyeLink(),
            },
            (16, 7),
            {
                'time': pl.Int64,
                'pupil': pl.Float64,
                'task': pl.Utf8,
                'screen_id': pl.Int64,
                'point_id': pl.Int64,
                'trial_id': pl.Int64,
                'pixel': pl.List(pl.Float64),
            },
            id='eyelink_asc_mono_definition',
        ),

        pytest.param(
            {
                'file': 'tests/files/eyelink_monocular_example.asc',
                'definition': ToyDatasetEyeLink(),
                'schema': {
                    'trial_id': pl.Int32,
                    'screen_id': pl.Int32,
                    'point_id': pl.Int32,
                    'task': pl.Utf8,
                },
            },
            (16, 7),
            {
                'time': pl.Int64,
                'pupil': pl.Float64,
                'task': pl.Utf8,
                'screen_id': pl.Int32,
                'point_id': pl.Int32,
                'trial_id': pl.Int32,
                'pixel': pl.List(pl.Float64),
            },
            id='eyelink_asc_mono_schema_overrides_definition',
        ),

        pytest.param(
            {
                'file': 'tests/files/eyelink_monocular_2khz_example.asc',
                'patterns': 'eyelink',
            },
            (16, 3),
            {
                'time': pl.Float64,
                'pupil': pl.Float64,
                'pixel': pl.List(pl.Float64),
            },
            id='eyelink_asc_mono_2khz_pattern_eyelink',
        ),

        pytest.param(
            {
                'file': 'tests/files/eyelink_monocular_no_dummy_example.asc',
                'patterns': 'eyelink',
                'encoding': 'latin1',
            },
            (297, 3),
            {
                'time': pl.Int64,
                'pupil': pl.Float64,
                'pixel': pl.List(pl.Float64),
            },
            id='eyelink_asc_mono_no_dummy_pattern_eyelink',
        ),

        pytest.param(
            {
                'file': 'tests/files/eyelink_monocular_no_dummy_example.asc',
                'patterns': 'eyelink',
                'definition': DatasetDefinition(
                    experiment=None,
                    custom_read_kwargs={'gaze': {'encoding': 'latin1'}},
                ),
            },
            (297, 3),
            {
                'time': pl.Int64,
                'pupil': pl.Float64,
                'pixel': pl.List(pl.Float64),
            },
            id='eyelink_asc_mono_no_dummy_pattern_eyelink_encoding_definition',
        ),

        pytest.param(
            {
                'file': 'tests/files/eyelink_monocular_no_dummy_example.asc',
                'patterns': 'eyelink',
                'definition': DatasetDefinition(
                    experiment=None,
                    custom_read_kwargs={'gaze': {'encoding': 'ascii'}},
                ),
                'encoding': 'latin1',
            },
            (297, 3),
            {
                'time': pl.Int64,
                'pupil': pl.Float64,
                'pixel': pl.List(pl.Float64),
            },
            id='eyelink_asc_mono_no_dummy_pattern_eyelink_encoding_overrides_definition',
        ),
        pytest.param(
            {
                'file': 'tests/files/eyelink_binocular_example.asc',
                'patterns': 'eyelink',
            },
            (368, 3),
            {
                'time': pl.Int64,
                'pixel': pl.List(pl.Float64),
                'pupil': pl.List(pl.Float64),
            },
            id='eyelink_asc_bino_pattern_eyelink',
        ),
    ],
)
def test_from_asc_has_shape_and_schema(kwargs, shape, schema):
    gaze = from_asc(**kwargs)

    assert gaze.samples.shape == shape
    assert dict(gaze.samples.schema) == schema


@pytest.mark.parametrize(
    ('kwargs', 'exception', 'message_prefix'),
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

        pytest.param(
            {
                'file': 'tests/files/eyelink_monocular_no_dummy_example.asc',
                'metadata_patterns': [
                    {'pattern': r'ENCODING TEST (?P<foobar>.+)'},
                ],
                'encoding': 'ascii',
            },
            UnicodeDecodeError,
            'ascii',
            id='eyelink_monocular_no_dummy_example_encoding_ascii',
        ),
    ],
)
def test_from_asc_raises_exception(kwargs, exception, message_prefix):
    with pytest.raises(exception) as excinfo:
        from_asc(**kwargs)

    msg = excinfo.value.args[0]
    assert msg.startswith(message_prefix)


@pytest.mark.parametrize(
    ('init_kwargs', 'expected_experiment'),
    [
        pytest.param(
            {'file': 'tests/files/eyelink_monocular_example.asc'},
            Experiment(
                screen=Screen(
                    width_px=1280,
                    height_px=1024,
                ),
                eyetracker=EyeTracker(
                    sampling_rate=1000.0,
                    left=True,
                    right=False,
                    model='EyeLink Portable Duo',
                    version='6.12',
                    vendor='EyeLink',
                    mount='Desktop',
                ),
            ),
            id='1khz',
        ),

        pytest.param(
            {
                'file': 'tests/files/eyelink_monocular_example.asc',
                'experiment': Experiment(
                    screen_width_cm=40, screen_height_cm=30, sampling_rate=1000,
                ),
            },
            Experiment(
                screen=Screen(
                    width_cm=40,
                    height_cm=30,
                    width_px=1280,
                    height_px=1024,
                ),
                eyetracker=EyeTracker(
                    sampling_rate=1000.0,
                    left=True,
                    right=False,
                    model='EyeLink Portable Duo',
                    version='6.12',
                    vendor='EyeLink',
                    mount='Desktop',
                ),
            ),
            id='1khz_experiment',
        ),

        pytest.param(
            {
                'file': 'tests/files/eyelink_monocular_example.asc',
                'definition': DatasetDefinition(
                    experiment=Experiment(
                        screen_width_cm=66, screen_height_cm=77, sampling_rate=1000,
                    ),
                ),
            },
            Experiment(
                screen=Screen(
                    width_cm=66,
                    height_cm=77,
                    width_px=1280,
                    height_px=1024,
                ),
                eyetracker=EyeTracker(
                    sampling_rate=1000.0,
                    left=True,
                    right=False,
                    model='EyeLink Portable Duo',
                    version='6.12',
                    vendor='EyeLink',
                    mount='Desktop',
                ),
            ),
            id='1khz_experiment_definition',
        ),

        pytest.param(
            {
                'file': 'tests/files/eyelink_monocular_example.asc',
                'definition': DatasetDefinition(
                    experiment=Experiment(
                        screen_width_cm=40, screen_height_cm=30, sampling_rate=1000,
                    ),
                ),
                'experiment': Experiment(
                    screen_width_cm=80, screen_height_cm=60, sampling_rate=1000,
                ),
            },
            Experiment(
                screen=Screen(
                    width_cm=80,
                    height_cm=60,
                    width_px=1280,
                    height_px=1024,
                ),
                eyetracker=EyeTracker(
                    sampling_rate=1000.0,
                    left=True,
                    right=False,
                    model='EyeLink Portable Duo',
                    version='6.12',
                    vendor='EyeLink',
                    mount='Desktop',
                ),
            ),
            id='1khz_experiment_overrides_definition',
        ),

        pytest.param(
            {'file': 'tests/files/eyelink_monocular_2khz_example.asc'},
            Experiment(
                screen=Screen(
                    width_px=1280,
                    height_px=1024,
                ),
                eyetracker=EyeTracker(
                    sampling_rate=2000.0,
                    left=True,
                    right=False,
                    model='EyeLink Portable Duo',
                    version='6.12',
                    vendor='EyeLink',
                    mount='Desktop',
                ),
            ),
            id='2khz',
        ),

        pytest.param(
            {
                'file': 'tests/files/eyelink_monocular_no_dummy_example.asc',
                'encoding': 'latin1',
            },
            Experiment(
                screen=Screen(
                    width_px=1920,
                    height_px=1080,
                ),
                eyetracker=EyeTracker(
                    sampling_rate=500.0,
                    left=True,
                    right=False,
                    model='EyeLink 1000 Plus',
                    version='5.50',
                    vendor='EyeLink',
                    mount='Desktop',
                ),
            ),
            id='500hz_no_dummy',
        ),
    ],
)
def test_from_asc_fills_in_experiment_metadata(init_kwargs, expected_experiment):
    gaze = from_asc(**init_kwargs)
    assert gaze.experiment == expected_experiment


@pytest.mark.parametrize(
    ('init_kwargs', 'expected_trial_columns'),
    [
        pytest.param(
            {
                'file': 'tests/files/eyelink_monocular_example.asc',
                'definition': ToyDatasetEyeLink(),
            },
            ['task', 'trial_id'],
            id='eyelink_asc_mono_definition',
        ),

        pytest.param(
            {
                'file': 'tests/files/eyelink_monocular_example.asc',
                'definition': ToyDatasetEyeLink(),
                'trial_columns': ['trial_id'],
            },
            ['trial_id'],
            id='eyelink_asc_mono_trial_columns_override_definition',
        ),

    ],
)
def test_from_asc_has_expected_trial_columns(init_kwargs, expected_trial_columns):
    gaze = from_asc(**init_kwargs)
    assert gaze.trial_columns == expected_trial_columns


@pytest.mark.parametrize(
    ('experiment_kwargs', 'issues'),
    [
        pytest.param(
            {
                'screen_width_px': 1920,
                'screen_height_px': 1080,
                'sampling_rate': 1000,
            },
            ['Screen resolution: (1920, 1080) != (1280.0, 1024.0)'],
            id='screen_resolution',
        ),
        pytest.param(
            {
                'eyetracker': EyeTracker(sampling_rate=500),
            },
            ['Sampling rate: 500 != 1000.0'],
            id='eyetracker_sampling_rate',
        ),
        pytest.param(
            {
                'eyetracker': EyeTracker(
                    left=False,
                    right=True,
                    sampling_rate=1000,
                    mount='Desktop',
                ),
            },
            [
                'Left eye tracked: False != True',
                'Right eye tracked: True != False',
            ],
            id='eyetracker_tracked_eye',
        ),
        pytest.param(
            {
                'eyetracker': EyeTracker(
                    vendor='Tobii',
                    model='Tobii Pro Spectrum',
                    version='1.0',
                    sampling_rate=1000,
                    left=True,
                    right=False,
                ),
            },
            [
                'Eye tracker vendor: Tobii != EyeLink',
                'Eye tracker model: Tobii Pro Spectrum != EyeLink Portable Duo',
                'Eye tracker software version: 1.0 != 6.12',
            ],
            id='eyetracker_vendor_model_version',
        ),
        pytest.param(
            {
                'eyetracker': EyeTracker(
                    mount='Remote',
                    sampling_rate=1000,
                    vendor='EyeLink',
                    model='EyeLink Portable Duo',
                    version='6.12',
                ),
            },
            ['Mount configuration: Remote != Desktop'],
            id='eyetracker_mount',
        ),
    ],
)
def test_from_asc_detects_mismatches_in_experiment_metadata(experiment_kwargs, issues):
    with pytest.raises(ValueError) as excinfo:
        from_asc(
            'tests/files/eyelink_monocular_example.asc',
            experiment=Experiment(**experiment_kwargs),
        )

    msg, = excinfo.value.args
    expected_msg = 'Experiment metadata does not match the metadata in the ASC file:\n'
    expected_msg += '\n'.join(f'- {issue}' for issue in issues)
    assert msg == expected_msg


@pytest.mark.parametrize(
    ('kwargs', 'expected_metadata'),
    [
        pytest.param(
            {
                'file': 'tests/files/eyelink_monocular_example.asc',
                'definition': ToyDatasetEyeLink(),
                'metadata_patterns': [
                    {'pattern': r'!V TRIAL_VAR SUBJECT_ID (?P<subject_id>-?\d+)'},
                ],
            },
            {
                'subject_id': '-1',
            },
            id='eyelink_asc_mono_subject_id_metadata_patterns',
        ),

        pytest.param(
            {
                'file': 'tests/files/eyelink_monocular_example.asc',
                'definition': ToyDatasetEyeLink(
                    trial_columns=None,
                    custom_read_kwargs={
                        'gaze': {
                            'metadata_patterns': [
                                {'pattern': r'!V TRIAL_VAR SUBJECT_ID (?P<subject_id>-?\d+)'},
                            ],
                        },
                    },
                ),
            },
            {
                'subject_id': '-1',
            },
            id='eyelink_asc_mono_subject_id_definition',
        ),

        pytest.param(
            {
                'file': 'tests/files/eyelink_monocular_example.asc',
                'definition': ToyDatasetEyeLink(
                    trial_columns=None,
                    custom_read_kwargs={
                        'gaze': {
                            'metadata_patterns': [
                                {'pattern': r'!V TRIAL_VAR SUBJECT_ID (?P<foobar>-?\d+)'},
                            ],
                        },
                    },
                ),
                'metadata_patterns': [
                    {'pattern': r'!V TRIAL_VAR SUBJECT_ID (?P<subject_id>-?\d+)'},
                ],
            },
            {
                'subject_id': '-1',
            },
            id='eyelink_asc_mono_subject_id_metadata_patterns_overrides_definition',
        ),

        pytest.param(
            {
                'file': 'tests/files/eyelink_monocular_no_dummy_example.asc',
                'metadata_patterns': [
                    {'pattern': r'ENCODING TEST (?P<foobar>.+)'},
                ],
                'encoding': 'utf8',
            },
            {
                'foobar': 'ÄÖÜ',
            },
            id='eyelink_monocular_no_dummy_example_encoding_utf8',
        ),

        pytest.param(
            {
                'file': 'tests/files/eyelink_monocular_no_dummy_example.asc',
                'metadata_patterns': [
                    {'pattern': r'ENCODING TEST (?P<foobar>.+)'},
                ],
                'encoding': 'latin1',
            },
            {
                'foobar': 'Ã\x84Ã\x96Ã\x9c',
            },
            id='eyelink_monocular_no_dummy_example_encoding_latin1',
        ),
    ],
)
def test_from_asc_has_expected_metadata(kwargs, expected_metadata):
    gaze = from_asc(**kwargs)

    for key, value in expected_metadata.items():
        assert key in gaze._metadata
        assert gaze._metadata[key] == value


@pytest.mark.parametrize(
    ('kwargs', 'expected_event_frame'),
    [
        pytest.param(
            {
                'file': 'tests/files/eyelink_monocular_example.asc',
                'events': False,
            },
            pl.from_dict(
                data={
                    'name': [],
                    'onset': [],
                    'offset': [],
                    'duration': [],
                },
                schema={
                    'name': pl.Utf8,
                    'onset': pl.Int64,
                    'offset': pl.Int64,
                    'duration': pl.Int64,
                },
            ),
            id='eyelink_asc_mono_without_events',
        ),
        pytest.param(
            {
                'file': 'tests/files/eyelink_monocular_example.asc',
                'events': True,
            },
            pl.from_dict(
                data={
                    'name': ['fixation_left_eyelink', 'saccade_left_eyelink', 'fixation_left_eyelink'],
                    'onset': [2154563, 2339227, 2339246],
                    'offset': [2154695, 2339245, 2339290],
                    'duration': [132, 18, 44],
                },
                schema={
                    'name': pl.Utf8,
                    'onset': pl.Int64,
                    'offset': pl.Int64,
                    'duration': pl.Int64,
                },
            ),
            id='eyelink_asc_mono_with_events',
        ),
        pytest.param(
            {
                'file': 'tests/files/eyelink_monocular_2khz_example.asc',
                'events': True,
            },
            pl.from_dict(
                data={
                    'name': ['fixation_left_eyelink', 'saccade_left_eyelink', 'fixation_left_eyelink'],
                    'onset': [2154563, 2339227, 2339246],
                    'offset': [2154695, 2339245, 2339290],
                    'duration': [132, 18, 44],
                },
                schema={
                    'name': pl.Utf8,
                    'onset': pl.Int64,
                    'offset': pl.Int64,
                    'duration': pl.Int64,
                },
            ),
            id='eyelink_asc_mono_2khz_with_events',
        ),
    ],
)
def test_from_asc_events(kwargs, expected_event_frame):
    gaze = from_asc(**kwargs)

    assert_frame_equal(gaze.events.frame, expected_event_frame, check_column_order=False)
