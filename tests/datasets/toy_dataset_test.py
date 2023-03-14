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
"""Test all functionality in pymovements.datasets.judo1000."""
from __future__ import annotations

from pathlib import Path

import pytest

from pymovements.datasets.toy_dataset import ToyDataset


@pytest.mark.parametrize(
    'init_kwargs, expected_paths',
    [
        pytest.param(
            {'root': '/data/set/path'},
            {
                'root': Path('/data/set/path/'),
                'path': Path('/data/set/path/ToyDataset'),
                'download': Path('/data/set/path/ToyDataset/downloads'),
            },
        ),
        pytest.param(
            {'root': '/data/set/path', 'dataset_dirname': '.'},
            {
                'root': Path('/data/set/path/'),
                'path': Path('/data/set/path/'),
                'download': Path('/data/set/path/downloads'),
            },
        ),
        pytest.param(
            {'root': '/data/set/path', 'dataset_dirname': 'dataset'},
            {
                'root': Path('/data/set/path/'),
                'path': Path('/data/set/path/dataset'),
                'download': Path('/data/set/path/dataset/downloads'),
            },
        ),
        pytest.param(
            {'root': '/data/set/path', 'downloads_dirname': 'custom_download_dirname'},
            {
                'root': Path('/data/set/path/'),
                'path': Path('/data/set/path/ToyDataset'),
                'download': Path('/data/set/path/ToyDataset/custom_download_dirname'),
            },
        ),
    ],
)
def test_paths(init_kwargs, expected_paths):
    dataset = ToyDataset(**init_kwargs)

    assert dataset.root == expected_paths['root']
    assert dataset.path == expected_paths['path']
    assert dataset.downloads_rootpath == expected_paths['download']
