# Copyright (c) 2025 The pymovements Project Authors
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
"""Tests deprecated utils.downloads."""
import re

import pytest

from pymovements import __version__
from pymovements.utils.downloads import download_and_extract_archive
from pymovements.utils.downloads import download_file


@pytest.mark.filterwarnings('ignore::DeprecationWarning')
@pytest.mark.parametrize('download_function', [download_and_extract_archive, download_file])
def test_download_file(download_function, tmp_path):
    url = 'https://github.com/aeye-lab/pymovements/archive/refs/tags/v0.4.0.tar.gz'
    filename = 'pymovements-0.4.0.tar.gz'

    download_function(url, tmp_path, filename)


@pytest.mark.parametrize('download_function', [download_and_extract_archive, download_file])
def test_archive_extract_deprecated(download_function, tmp_path):
    url = 'https://github.com/aeye-lab/pymovements/archive/refs/tags/v0.4.0.tar.gz'
    filename = 'pymovements-0.4.0.tar.gz'

    with pytest.raises(DeprecationWarning):
        download_function(url, tmp_path, filename)


@pytest.mark.parametrize('download_function', [download_and_extract_archive, download_file])
def test_archive_extract_removed(download_function, tmp_path):
    url = 'https://github.com/aeye-lab/pymovements/archive/refs/tags/v0.4.0.tar.gz'
    filename = 'pymovements-0.4.0.tar.gz'

    with pytest.raises(DeprecationWarning) as info:
        download_function(url, tmp_path, filename)

    regex = re.compile(r'.*will be removed in v(?P<version>[0-9]*[.][0-9]*[.][0-9]*)[.)].*')

    msg = info.value.args[0]
    remove_version = regex.match(msg).groupdict()['version']
    current_version = __version__.split('+')[0]
    assert current_version < remove_version, (
        f'utils/downloads.py was planned to be removed in v{remove_version}. '
        f'Current version is v{current_version}.'
    )
