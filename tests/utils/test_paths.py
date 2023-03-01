# Copyright (c) 2022-2023 The pymovements Project Authors
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
"""
Test pymovements paths.
"""
import pathlib
import re

import pytest

from pymovements.utils.paths import get_filepaths


def test_get_filepaths_mut_excl_extension_and_regex_error():
    """
    Test mutually exclusive extension and regex in get_filepaths.
    """
    with pytest.raises(ValueError) as excinfo:
        get_filepaths(path='tmp', extension='extension', regex=re.compile('regex'))
    msg, = excinfo.value.args
    assert msg == 'extension and regex are mutually exclusive'


@pytest.mark.parametrize(
    ('extension', 'regex', 'expected_paths'),
    [
        pytest.param(
            None,
            re.compile('regex'),
            [],
            id='regex empty dir',
        ),
        pytest.param(
            'extension',
            None,
            [],
            id='extension empty dir',
        ),
    ],
)
def test_get_filepaths_empty_directory(
        extension,
        regex,
        expected_paths,
):
    """
    Test mutually exclusive extension and regex in `get_filepaths`.
    """
    assert get_filepaths(path='tmp', extension=extension, regex=regex) == expected_paths


def create_directory(tmp_path, sub_dirs, files):
    for sub_dir in sub_dirs:
        dir_path = tmp_path / sub_dir
        dir_path.mkdir()
        for file_name in files:
            file_path = dir_path / file_name
            file_path.write_text('test')


@pytest.mark.parametrize(
    ('extension', 'regex', 'sub_dirs', 'files', 'expected_paths'),
    [
        pytest.param(
            '.txt',
            None,
            ['tmp_dir'],
            ['foo.txt', 'bar.py'],
            ['tmp_dir/foo.txt'],
            id='extension regex get filepaths list',
        ),
        pytest.param(
            None,
            re.compile('foo'),
            ['tmp_dir'],
            ['foo.txt', 'foo.py', 'bar.py'],
            ['tmp_dir/foo.txt', 'tmp_dir/foo.py'],
            id='regex get filepaths list',
        ),
    ],
)
def test_get_filepaths_is_dir(
        extension,
        regex,
        tmp_path,
        sub_dirs,
        files,
        expected_paths,
):
    """
    Test `get_filepaths` list creation.
    """
    create_directory(tmp_path, sub_dirs, files)
    ret = get_filepaths(path=tmp_path, extension=extension, regex=regex)
    expected_list = [tmp_path / pathlib.Path(expected_path) for expected_path in expected_paths]
    assert sorted(ret) == sorted(expected_list)
