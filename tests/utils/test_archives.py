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
Test pymovements utils archives.
"""
import bz2
import gzip
import lzma
import pathlib
import tarfile
import zipfile

import pytest

from pymovements.utils.archives import _ZIP_COMPRESSION_MAP
from pymovements.utils.archives import extract_archive


def test_extract_archive_wrong_suffix():
    """
    Test unsupported suffix for extract_archive()
    """
    with pytest.raises(RuntimeError) as excinfo:
        extract_archive(pathlib.Path('test.jpg'))
    msg, = excinfo.value.args
    assert msg == """Unsupported compression or archive type: '.jpg'.
Supported suffixes are: '['.bz2', '.gz', '.tar', '.tbz', '.tbz2', '.tgz', '.xz', '.zip']'."""


def test_detect_file_type_no_suffixes():
    """
    Test extract_archive() for no files with suffix
    """
    with pytest.raises(RuntimeError) as excinfo:
        extract_archive(pathlib.Path('test'))
    msg, = excinfo.value.args
    assert msg == "File 'test' has no suffixes that could be used to "\
        'detect the archive type or compression.'


@pytest.fixture(
    name='archive',
    params=[
        (None, 'tar'),
        (None, 'zip'),
        ('bz2', None),
        ('gz', None),
        ('tbz', None),
        ('tbz2', None),
        ('tgz', None),
        ('xz', None),
        ('bz2', 'tar'),
        ('bz2', 'zip'),
        ('gz', 'tar'),
        ('xz', 'tar'),
        ('xz', 'zip'),
    ],
    ids=[
        'tar_archive',
        'zip_archive',
        'bz2_compressed_archive',
        'gz_compressed_archive',
        'tbz_compressed_archive',
        'tbz2_compressed_archive',
        'tgz_compressed_archive',
        'xz_compressed_archive',
        'bz2_compressed_tar_archive',
        'bz2_compressed_zip_archive',
        'gz_compressed_tar_archive',
        'xz_compressed_tar_archive',
        'xz_compressed_zip_archive',
    ],
)
def fixture_archive(request, tmp_path):
    rootpath = tmp_path
    compression, extension = request.param

    # write tmp filepath
    filepath = rootpath / 'test.file'
    if extension in ['zip', 'tar']:
        filepath.write_text('test')
    if compression in ['bz2', 'xz', 'tbz', 'tbz2', 'tgz', 'gz'] and extension is None:
        filepath.write_bytes(b'test')

    # declare archive path
    if compression is None:
        archive_path = rootpath / f'test.{extension}'
    elif compression is not None and extension is None:
        archive_path = rootpath / f'test.{compression}'
    elif compression is not None and extension is not None:
        archive_path = rootpath / f'test.{extension}.{compression}'

    if compression is None and extension == 'zip':
        with zipfile.ZipFile(archive_path, 'w') as zip_open:
            zip_open.write(filepath)
        yield archive_path

    elif compression is not None and extension == 'zip':
        comp_type = _ZIP_COMPRESSION_MAP[f'.{compression}']
        with zipfile.ZipFile(archive_path, 'w', compression=comp_type) as zip_open:
            zip_open.write(filepath)
        yield archive_path

    elif compression is None and extension == 'tar':
        with tarfile.TarFile.open(archive_path, 'w') as fp:
            fp.add(filepath)
        yield archive_path

    elif (
        (compression is not None and extension == 'tar') or
        (compression in ['tbz', 'tbz2', 'tgz'])
    ):
        if compression in ['tbz', 'tbz2']:
            compression = 'bz2'
        if compression in ['tgz']:
            compression = 'gz'
        with tarfile.TarFile.open(archive_path, f'w:{compression}') as fp:
            fp.add(filepath)
        yield archive_path

    elif compression == 'bz2' and extension is None:
        with bz2.open(archive_path, 'wb') as fp:
            fp.write(filepath.read_bytes())
        yield archive_path

    elif compression == 'gz' and extension is None:
        with gzip.open(archive_path, 'wb') as fp:
            fp.write(filepath.read_bytes())
        yield archive_path

    elif compression == 'xz' and extension is None:
        with lzma.open(archive_path, 'wb') as fp:
            fp.write(filepath.read_bytes())
        yield archive_path

    else:
        raise ValueError(f'{request.param} not supported for archive fixture')


@pytest.fixture(
    name='unsupported_archive',
    params=[
        ('xz', 'jpg'),
    ],
    ids=[
        'xz_compressed_unsupported_archive',
    ],
)
def fixture_unsupported_archive(request, tmp_path):
    rootpath = tmp_path
    compression, extension = request.param

    # write tmp filepath
    filepath = rootpath / 'test.file'
    filepath.write_text('test')
    archive_path = rootpath / f'test.{extension}.{compression}'
    comp_type = _ZIP_COMPRESSION_MAP[f'.{compression}']
    with zipfile.ZipFile(archive_path, 'w', compression=comp_type) as zip_open:
        zip_open.write(filepath)
    yield archive_path


@pytest.mark.parametrize(
    'recursive',
    [
        pytest.param(False, id='recursive_false'),
        pytest.param(True, id='recursive_true'),
    ],
)
@pytest.mark.parametrize(
    'remove_finished',
    [
        pytest.param(False, id='remove_finished_false'),
        pytest.param(True, id='remove_finished_true'),
    ],
)
def test_extract_archive_destination_path_None(recursive, remove_finished, archive):
    extract_archive(
        source_path=archive,
        destination_path=None,
        recursive=recursive,
        remove_finished=remove_finished,
    )


@pytest.mark.parametrize(
    'recursive',
    [
        pytest.param(False, id='recursive_false'),
        pytest.param(True, id='recursive_true'),
    ],
)
@pytest.mark.parametrize(
    'remove_finished',
    [
        pytest.param(False, id='remove_finished_false'),
        pytest.param(True, id='remove_finished_true'),
    ],
)
def test_extract_unsupported_archive_destination_path_None(
    recursive, remove_finished, unsupported_archive,
):
    with pytest.raises(RuntimeError) as excinfo:
        extract_archive(
            source_path=unsupported_archive,
            destination_path=None,
            recursive=recursive,
            remove_finished=remove_finished,
        )
    msg, = excinfo.value.args
    assert msg == """Unsupported archive type: '.jpg'.
Supported suffixes are: '['.tar', '.zip']'."""


@pytest.mark.parametrize(
    'recursive',
    [
        pytest.param(False, id='recursive_false'),
        pytest.param(True, id='recursive_true'),
    ],
)
@pytest.mark.parametrize(
    'remove_finished',
    [
        pytest.param(False, id='remove_finished_false'),
        pytest.param(True, id='remove_finished_true'),
    ],
)
def test_extract_archive_destination_path_not_None(recursive, remove_finished, archive, tmp_path):
    destination_path = tmp_path / pathlib.Path('tmpfoo')
    extract_archive(
        source_path=archive,
        destination_path=destination_path,
        recursive=recursive,
        remove_finished=remove_finished,
    )


@pytest.mark.parametrize(
    'recursive',
    [
        pytest.param(False, id='recursive_false'),
        pytest.param(True, id='recursive_true'),
    ],
)
@pytest.mark.parametrize(
    'remove_finished',
    [
        pytest.param(False, id='remove_finished_false'),
        pytest.param(True, id='remove_finished_true'),
    ],
)
def test_extract_unnsupported_archive_destination_path_not_None(
        recursive, remove_finished, unsupported_archive, tmp_path,
):
    destination_path = tmp_path / pathlib.Path('tmpfoo')
    with pytest.raises(RuntimeError) as excinfo:
        extract_archive(
            source_path=unsupported_archive,
            destination_path=destination_path,
            recursive=recursive,
            remove_finished=remove_finished,
        )
    msg, = excinfo.value.args
    assert msg == """Unsupported archive type: '.jpg'.
Supported suffixes are: '['.tar', '.zip']'."""
