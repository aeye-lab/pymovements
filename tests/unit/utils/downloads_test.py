import re

import pytest

from pymovements.utils.downloads import download_and_extract_archive
from pymovements.utils.downloads import download_file
from pymovements import __version__


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

