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
"""Test dataset definition."""
from dataclasses import dataclass

import pytest
import yaml

from pymovements import DatasetDefinition
from pymovements import DatasetLibrary
from pymovements import Experiment


@pytest.mark.parametrize(
    'init_kwargs',
    [
        pytest.param(
            {'name': 'A'},
            id='name_only',
        ),
        pytest.param(
            {'name': 'A', 'experiment': Experiment(sampling_rate=1000)},
            id='name_and_experiment',
        ),
    ],
)
def test_dataset_definition_is_equal(init_kwargs):
    definition1 = DatasetDefinition(**init_kwargs)
    definition2 = DatasetDefinition(**init_kwargs)

    assert definition1 == definition2


@pytest.mark.parametrize(
    ('definition', 'expected_dict'),
    [
        pytest.param(
            DatasetDefinition(
                name='Example',
                long_name='Example',
                has_files={
                    'gaze': False,
                    'precomputed_events': False,
                    'precomputed_reading_measures': False,
                },
            ),
            {
                'name': 'Example',
                'long_name': 'Example',
                'has_files': {
                    'gaze': False,
                    'precomputed_events': False,
                    'precomputed_reading_measures': False,
                },
                'acceleration_columns': None,
                'column_map': {},
                'custom_read_kwargs': {},
                'distance_column': None,
                'experiment': {
                    'eyetracker': {
                        'left': None,
                        'model': None,
                        'mount': None,
                        'right': None,
                        'sampling_rate': None,
                        'vendor': None,
                        'version': None,
                    },
                    'screen': {
                        'distance_cm': None,
                        'height_cm': None,
                        'height_px': None,
                        'origin': 'upper left',
                        'width_cm': None,
                        'width_px': None,
                    },
                },
                'extract': {},
                'filename_format': {},
                'filename_format_schema_overrides': {},
                'mirrors': {},
                'pixel_columns': None,
                'position_columns': None,
                'resources': {},
                'time_column': None,
                'time_unit': 'ms',
                'trial_columns': None,
                'velocity_columns': None,
            },
            id='no_experiment',
        ),
        pytest.param(
            DatasetDefinition(
                name='Example',
                long_name='Example',
                has_files={
                    'gaze': False,
                    'precomputed_events': False,
                    'precomputed_reading_measures': False,
                },
                experiment=Experiment(
                    screen_width_px=1280,
                    screen_height_px=1024,
                    screen_width_cm=38.2,
                    screen_height_cm=30.2,
                    distance_cm=60,
                    origin='center',
                    sampling_rate=2000,
                ),
            ),
            {
                'name': 'Example',
                'long_name': 'Example',
                'has_files': {
                    'gaze': False,
                    'precomputed_events': False,
                    'precomputed_reading_measures': False,
                },
                'acceleration_columns': None,
                'column_map': {},
                'custom_read_kwargs': {},
                'distance_column': None,
                'experiment': {
                    'eyetracker': {
                        'left': None,
                        'model': None,
                        'mount': None,
                        'right': None,
                        'sampling_rate': 2000,
                        'vendor': None,
                        'version': None,
                    },
                    'screen': {
                        'distance_cm': 60,
                        'height_cm': 30.2,
                        'height_px': 1024,
                        'origin': 'center',
                        'width_cm': 38.2,
                        'width_px': 1280,
                    },
                },
                'extract': {},
                'filename_format': {},
                'filename_format_schema_overrides': {},
                'mirrors': {},
                'pixel_columns': None,
                'position_columns': None,
                'resources': {},
                'time_column': None,
                'time_unit': 'ms',
                'trial_columns': None,
                'velocity_columns': None,
            },
            id='experiment',
        ),
    ],
)
def test_dataset_definition_to_dict_expected(definition, expected_dict):
    assert definition.to_dict() == expected_dict


@pytest.mark.parametrize(
    ('exclude_private', 'expected_dict'),
    [
        pytest.param(
            True,
            {
                'name': 'MyDatasetDefinition',
                'long_name': None,
                'has_files': {},
                'acceleration_columns': None,
                'column_map': {},
                'custom_read_kwargs': {},
                'distance_column': None,
                'experiment': {
                    'eyetracker': {
                        'left': None,
                        'model': None,
                        'mount': None,
                        'right': None,
                        'sampling_rate': None,
                        'vendor': None,
                        'version': None,
                    },
                    'screen': {
                        'distance_cm': None,
                        'height_cm': None,
                        'height_px': None,
                        'origin': 'upper left',
                        'width_cm': None,
                        'width_px': None,
                    },
                },
                'extract': {},
                'filename_format': {},
                'filename_format_schema_overrides': {},
                'mirrors': {},
                'pixel_columns': None,
                'position_columns': None,
                'resources': {},
                'time_column': None,
                'time_unit': 'ms',
                'trial_columns': None,
                'velocity_columns': None,
            },
            id='True',
        ),

        pytest.param(
            False,
            {
                'name': 'MyDatasetDefinition',
                'long_name': None,
                '_foobar': 'test',
                'has_files': {},
                'acceleration_columns': None,
                'column_map': {},
                'custom_read_kwargs': {},
                'distance_column': None,
                'experiment': {
                    'eyetracker': {
                        'left': None,
                        'model': None,
                        'mount': None,
                        'right': None,
                        'sampling_rate': None,
                        'vendor': None,
                        'version': None,
                    },
                    'screen': {
                        'distance_cm': None,
                        'height_cm': None,
                        'height_px': None,
                        'origin': 'upper left',
                        'width_cm': None,
                        'width_px': None,
                    },
                },
                'extract': {},
                'filename_format': {},
                'filename_format_schema_overrides': {},
                'mirrors': {},
                'pixel_columns': None,
                'position_columns': None,
                'resources': {},
                'time_column': None,
                'time_unit': 'ms',
                'trial_columns': None,
                'velocity_columns': None,
            },
            id='False',
        ),
    ],
)
def test_dataset_definition_to_dict_exclude_private_expected(exclude_private, expected_dict):
    @dataclass
    class MyDatasetDefinition(DatasetDefinition):
        name: str = 'MyDatasetDefinition'
        _foobar: str = 'test'

    definition = MyDatasetDefinition()

    assert definition.to_dict(exclude_private=exclude_private) == expected_dict


@pytest.mark.parametrize(
    ('definition'),
    [
        pytest.param(
            DatasetDefinition(
                name='Example',
                has_files={
                    'gaze': False,
                    'precomputed_events': False,
                    'precomputed_reading_measures': False,
                },
            ),
            id='no_exp',
        ),

        pytest.param(
            DatasetDefinition(
                name='Example',
                has_files={
                    'gaze': False,
                    'precomputed_events': False,
                    'precomputed_reading_measures': False,
                },
                experiment=Experiment(
                    screen_width_px=1280,
                    screen_height_px=1024,
                    screen_width_cm=38.2,
                    screen_height_cm=30.2,
                    distance_cm=60,
                    origin='center',
                    sampling_rate=2000,
                ),
            ),
            id='no_exp',
        ),
    ],
)
def test_dataset_definition_to_yaml_equal_dicts(definition, tmp_path):
    tmp_file = tmp_path / 'tmp.yaml'

    definition.to_yaml(tmp_file)

    with open(tmp_file, encoding='utf-8') as f:
        yaml_dict = yaml.safe_load(f)

    assert definition.to_dict() == yaml_dict


def test_write_yaml_already_existing_dataset_definition_w_tuple_screen(tmp_path):
    tmp_file = tmp_path / 'tmp.yaml'
    definition = DatasetLibrary.get('ToyDatasetEyeLink')
    definition.to_yaml(tmp_file)

    with open(tmp_file, encoding='utf-8') as f:
        yaml.safe_load(f)

    assert DatasetDefinition.from_yaml(tmp_file) == definition


def test_check_equality_of_load_from_yaml_and_load_from_dictionary_dump(tmp_path):
    dictionary_tmp_file = tmp_path / 'dictionary.yaml'
    yaml_encoding = {
        'name': 'Example',
        'has_files': {
            'gaze': False,
            'precomputed_events': False,
            'precomputed_reading_measures': False,
        },
    }

    with open(dictionary_tmp_file, 'w', encoding='utf-8') as f:
        yaml.safe_dump(yaml_encoding, f)

    yaml_definition = DatasetDefinition.from_yaml(dictionary_tmp_file)

    expected_definition = DatasetDefinition(
        name='Example',
        has_files={
            'gaze': False,
            'precomputed_events': False,
            'precomputed_reading_measures': False,
        },
    )

    assert yaml_definition == expected_definition


def test_dataset_to_dict_exclude_none():
    definition = DatasetDefinition(
        name='Example',
        trial_columns=[],
    )

    dict_default = definition.to_dict()
    assert 'long_name' not in dict_default
    assert 'trial_columns' not in dict_default
    assert 'has_files' not in dict_default

    dict_non_default = definition.to_dict(exclude_none=False)
    assert 'long_name' in dict_non_default
    assert 'trial_columns' in dict_non_default
    assert 'has_files' in dict_non_default
