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
import hashlib
import os.path
import re

import pytest

from pymovements import __version__
from pymovements.utils.downloads import download_and_extract_archive
from pymovements.utils.downloads import download_file


@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_download_and_extract_archive(tmp_path):
    url = 'https://github.com/aeye-lab/pymovements/archive/refs/tags/v0.4.0.tar.gz'
    _download_filename = 'pymovements-0.4.0.tar.gz'
    md5 = '52bbf03a7c50ee7152ccb9d357c2bb30'
    extract_dirpath = tmp_path / 'extracted'

    download_and_extract_archive(
        url,
        tmp_path,
        _download_filename,
        extract_dirpath,
        md5,
        remove_top_level=False,
    )

    assert extract_dirpath.exists()
    assert (extract_dirpath / 'pymovements-0.4.0').exists()
    assert (extract_dirpath / 'pymovements-0.4.0' / 'README.md').exists()

    archive_path = tmp_path / _download_filename
    assert archive_path.exists()
    with open(archive_path, 'rb') as f:
        file_bytes = f.read()
        assert hashlib.md5(file_bytes).hexdigest() == md5


@pytest.mark.filterwarnings('ignore::DeprecationWarning')
@pytest.mark.parametrize(
    'verbose',
    [
        pytest.param(False, id='verbose_false'),
        pytest.param(True, id='verbose_true'),
    ],
)
def test_download_and_extract_archive_extract_dirpath_None(tmp_path, capsys, verbose):
    url = 'https://github.com/aeye-lab/pymovements/archive/refs/tags/v0.4.0.tar.gz'
    _download_filename = 'pymovements-0.4.0.tar.gz'
    md5 = '52bbf03a7c50ee7152ccb9d357c2bb30'
    extract_dirpath = None

    # extract first time
    download_and_extract_archive(
        url, tmp_path, _download_filename, extract_dirpath, md5, verbose=verbose,
    )
    out, _ = capsys.readouterr()

    if verbose:
        assert out == 'Downloading '\
            'https://github.com/aeye-lab/pymovements/archive/refs/tags/v0.4.0.tar.gz'\
            f" to {os.path.join(tmp_path, 'pymovements-0.4.0.tar.gz')}\n" \
            f'Checking integrity of pymovements-0.4.0.tar.gz\n'\
            f'Extracting pymovements-0.4.0.tar.gz to {tmp_path}\n'
    else:
        assert out == ''

    # extract second time to test already downloaded and verified file
    download_and_extract_archive(
        url, tmp_path, _download_filename, extract_dirpath, md5, verbose=verbose,
    )
    out, _ = capsys.readouterr()

    if verbose:
        assert out == f'Using already downloaded and verified file: '\
            f"{os.path.join(tmp_path, 'pymovements-0.4.0.tar.gz')}"\
            f'\nExtracting pymovements-0.4.0.tar.gz to {tmp_path}\n'
    else:
        assert out == ''


@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_download_and_extract_archive_invalid_md5(tmp_path):
    url = 'https://github.com/aeye-lab/pymovements/archive/refs/tags/v0.4.0.tar.gz'
    _download_filename = 'pymovements-0.4.0.tar.gz'
    md5 = '00000000000000000000000000000000'
    extract_dirpath = tmp_path / 'extracted'

    with pytest.raises(RuntimeError) as excinfo:
        download_and_extract_archive(url, tmp_path, _download_filename, extract_dirpath, md5)

    msg, = excinfo.value.args
    assert msg == f"File {os.path.join(tmp_path, 'pymovements-0.4.0.tar.gz')} "\
        'not found or download corrupted.'


@pytest.mark.filterwarnings('ignore::DeprecationWarning')
@pytest.mark.parametrize('download_function', [download_and_extract_archive, download_file])
def test_deprecated_download_function(download_function, tmp_path):
    url = 'https://github.com/aeye-lab/pymovements/archive/refs/tags/v0.4.0.tar.gz'
    filename = 'pymovements-0.4.0.tar.gz'

    download_function(url, tmp_path, filename)


@pytest.mark.parametrize('download_function', [download_and_extract_archive, download_file])
def test_is_download_function_deprecated(download_function, tmp_path):
    url = 'https://github.com/aeye-lab/pymovements/archive/refs/tags/v0.4.0.tar.gz'
    filename = 'pymovements-0.4.0.tar.gz'

    with pytest.raises(DeprecationWarning):
        download_function(url, tmp_path, filename)


@pytest.mark.parametrize('download_function', [download_and_extract_archive, download_file])
def test_is_download_function_removed(download_function, tmp_path):
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
