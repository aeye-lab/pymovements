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

import pytest
import yaml

from pymovements import DatasetDefinition
from pymovements import DatasetLibrary
from pymovements import Experiment


def test_dataset_definition_to_yaml_w_experiment(tmp_path):
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
        experiment: Experiment = Experiment(
            screen_width_px=1280,
            screen_height_px=1024,
            screen_width_cm=38.2,
            screen_height_cm=30.2,
            distance_cm=60,
            origin='center',
            sampling_rate=2000,
        )

    dataset = TestDatasetDefinition()
    dataset.to_yaml(tmp_file)

    with open(tmp_file, encoding='utf-8') as f:
        yaml_dict = yaml.safe_load(f)

    # hack until #919 resolved
    dataset_dict = asdict(dataset)
    dataset_experiment_dict = dataset_dict.pop('experiment').__dict__
    yaml_experiment = yaml_dict.pop('experiment')
    dataset_screen_dict = dataset_experiment_dict.pop('screen').__dict__
    assert dataset_dict == yaml_dict
    for key in yaml_experiment:
        if key.startswith('screen_'):
            key = key.strip('screen_')
        if key == 'sampling_rate':
            key = f'_{key}'

        assert key in (dataset_experiment_dict | dataset_screen_dict)


@pytest.mark.xfail(reason='#991')
def test_write_yaml_already_existing_dataset_definition_w_tuple(tmp_path):
    tmp_file = tmp_path / 'tmp.yaml'
    dataset = DatasetLibrary.get('ToyDatasetEyeLink')
    dataset.to_yaml(tmp_file)
    assert tmp_file.is_file()
    with open(tmp_file, encoding='utf-8') as f:
        written_file = yaml.safe_load(f)
    assert written_file == asdict(dataset)


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

    yaml_dataset = DatasetDefinition().from_yaml(dictionary_tmp_file)

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

    dataset = TestDatasetDefinition()

    # hack to compare the dataset definitions
    # change to dataset == yaml_dataset after #919 is resolved
    assert dataset.__dict__ == yaml_dataset.__dict__
