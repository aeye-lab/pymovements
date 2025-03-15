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
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field

import yaml

from pymovements import DatasetDefinition
from pymovements import DatasetLibrary
from pymovements import Experiment


def test_dataset_definition_to_yaml_equal_dicts_no_exp(tmp_path):
    tmp_file = tmp_path / 'tmp.yaml'

    @dataclass
    class TestDatasetDefinition(DatasetDefinition):
        name: str = 'Example'
        has_files: dict[str, bool] = field(
            default_factory=lambda: {
                'gaze': False,
                'precomputed_events': False,
                'precomputed_reading_measures': False,
            },
        )

    definition = TestDatasetDefinition()
    definition.to_yaml(tmp_file)

    with open(tmp_file, encoding='utf-8') as f:
        yaml_dict = yaml.safe_load(f)

    definition_dict = asdict(definition)
    definition_dict['experiment'] = definition_dict['experiment'].to_dict()
    assert definition_dict == yaml_dict


def test_dataset_definition_to_yaml_equal_dicts(tmp_path):
    tmp_file = tmp_path / 'tmp.yaml'

    @dataclass
    class TestDatasetDefinition(DatasetDefinition):
        name: str = 'Example'
        has_files: dict[str, bool] = field(
            default_factory=lambda: {
                'gaze': False,
                'precomputed_events': False,
                'precomputed_reading_measures': False,
            },
        )
        experiment: Experiment = field(
            default_factory=lambda: Experiment(
                screen_width_px=1280,
                screen_height_px=1024,
                screen_width_cm=38.2,
                screen_height_cm=30.2,
                distance_cm=60,
                origin='center',
                sampling_rate=2000,
            ),
        )

    definition = TestDatasetDefinition()
    definition.to_yaml(tmp_file)

    with open(tmp_file, encoding='utf-8') as f:
        yaml_dict = yaml.safe_load(f)

    definition_dict = asdict(definition)
    definition_dict['experiment'] = definition_dict['experiment'].to_dict()
    assert definition_dict == yaml_dict


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

    @dataclass
    class TestDatasetDefinition(DatasetDefinition):
        name: str = 'Example'
        has_files: dict[str, bool] = field(
            default_factory=lambda: {
                'gaze': False,
                'precomputed_events': False,
                'precomputed_reading_measures': False,
            },
        )

    definition = TestDatasetDefinition()

    assert definition.__dict__ == yaml_definition.__dict__
