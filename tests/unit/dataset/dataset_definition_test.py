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

import pytest
import yaml

import pymovements as pm


# can be removed when `test_write_yaml_already_existing_with_values` passes
def test_write_yaml_already_existing(tmp_path):
    tmp_file = tmp_path / 'tmp.yaml'
    dataset = pm.DatasetLibrary.get('ToyDatasetEyeLink')
    dataset.to_yaml(tmp_file)
    with open(tmp_file, encoding='utf-8') as f:
        written_file = yaml.safe_load(f)

    for key in asdict(dataset).keys():
        assert key in written_file


def test_write_yaml_already_existing_no_experiment(tmp_path):
    tmp_file = tmp_path / 'tmp.yaml'
    dataset = pm.DatasetLibrary.get('EMTeC')
    dataset.to_yaml(tmp_file)
    with open(tmp_file, encoding='utf-8') as f:
        written_file = yaml.safe_load(f)

    for key in asdict(dataset).keys():
        assert key in written_file


@pytest.mark.xfail(reason='#991')
def test_write_yaml_already_existing_with_values(tmp_path):
    tmp_file = tmp_path / 'tmp.yaml'
    dataset = pm.DatasetLibrary.get('ToyDatasetEyeLink')
    dataset.to_yaml(tmp_file)
    assert tmp_file.is_file()
    with open(tmp_file, encoding='utf-8') as f:
        written_file = yaml.safe_load(f)
    assert written_file == asdict(dataset)


def test_write_yaml_new(tmp_path):
    tmp_file = tmp_path / 'tmp.yaml'
    yaml_encoding = {
        'name': 'Example',
        'has_files': {
            'gaze': 'false',
            'precomputed_events': 'false',
            'precomputed_reading_measures': 'false',
        },
    }

    with open(tmp_file, 'w', encoding='utf-8') as f:
        yaml.safe_dump(yaml_encoding, f)
    dataset = pm.DatasetDefinition().from_yaml(tmp_file)
    dataset.to_yaml(tmp_file)

    # test initial dictionary definition is subset of written dictionary definition
    # (default values) ommited
    assert all(
        (item in asdict(dataset).items() or not item)
        for item in yaml_encoding.items()
    )
