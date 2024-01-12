# Copyright (c) 2023-2024 The pymovements Project Authors
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
"""Test all functionality in pymovements.dataset.sb_sat."""
from pathlib import Path

import pytest

import pymovements as pm


@pytest.mark.parametrize(
    'init_path, expected_paths',
    [
        pytest.param(
            '/data/set/path',
            {
                'root': Path('/data/set/path/'),
                'dataset': Path('/data/set/path/'),
                'download': Path('/data/set/path/downloads'),
            },
        ),
        pytest.param(
            pm.DatasetPaths(root='/data/set/path'),
            {
                'root': Path('/data/set/path/'),
                'dataset': Path('/data/set/path/SBSAT'),
                'download': Path('/data/set/path/SBSAT/downloads'),
            },
        ),
        pytest.param(
            pm.DatasetPaths(root='/data/set/path', dataset='.'),
            {
                'root': Path('/data/set/path/'),
                'dataset': Path('/data/set/path/'),
                'download': Path('/data/set/path/downloads'),
            },
        ),
        pytest.param(
            pm.DatasetPaths(root='/data/set/path', dataset='dataset'),
            {
                'root': Path('/data/set/path/'),
                'dataset': Path('/data/set/path/dataset'),
                'download': Path('/data/set/path/dataset/downloads'),
            },
        ),
        pytest.param(
            pm.DatasetPaths(root='/data/set/path', downloads='custom_downloads'),
            {
                'root': Path('/data/set/path/'),
                'dataset': Path('/data/set/path/SBSAT'),
                'download': Path('/data/set/path/SBSAT/custom_downloads'),
            },
        ),
    ],
)
def test_paths(init_path, expected_paths):
    dataset = pm.Dataset(pm.datasets.SBSAT, path=init_path)

    assert dataset.paths.root == expected_paths['root']
    assert dataset.path == expected_paths['dataset']
    assert dataset.paths.dataset == expected_paths['dataset']
    assert dataset.paths.downloads == expected_paths['download']
