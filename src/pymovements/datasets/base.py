from __future__ import annotations
from pathlib import Path
from urllib.error import URLError
from typing import Any
from typing import Callable
from typing import IO
from typing import Iterator
import hashlib
import sys
import urllib.request

from tqdm.auto import tqdm


USER_AGENT: str = 'aeye-lab/pymovements'


class Dataset:
    """

    """
    pass


class PublicDataset(Dataset):
    # TODO: add abstractmethod decorator
    mirrors = None
    resources = None

    def __init__(
        self,
        root: str,
        download: bool = False,
    ):
        self.root = Path(root)

        if download:
            self.download()

    def download(self):
        if not self.mirrors:
            raise ValueError("no mirrors defined for dataset")

        if not self.resources:
            raise ValueError("no resources defined for datasaet")

        self.raw_dirpath.mkdir(parents=True, exist_ok=True)

        for resource in self.resources:
            for mirror in self.mirrors:

                url = f'{mirror}{resource["path"]}'

                try:
                    print(f"Downloading {url}")
                    download_and_extract_archive(
                        url=url,
                        download_root=self.raw_dirpath,
                        filename=resource['filename'],
                        md5=resource['md5'],
                    )

                except URLError as error:
                    print(f"Failed to download (trying next):\n{error}")
                    # downloading the resource, try next mirror
                    continue
                    # TODO: check that at least one mirror was successful

                # downloading the resource was successful, we don't need to try another mirror
                break


    @property
    def dirpath(self) -> Path:
        return self.root / self.__class__.__name__.lower()

    @property
    def raw_dirpath(self) -> Path:
        return self.dirpath / "raw"


# TODO: move all these static methods to a utils module

def download_and_extract_archive(
    url: str,
    download_root: Path,
    extract_root: Path | None = None,
    filename: str | None = None,
    md5: str | None = None,
    remove_after_extract: bool = False,
):
    archive_path = download_url(
        url=url,
        rootpath=download_root,
        filename=filename,
        md5=md5,
    )

    print(f"Extracting {archive_path.name} to {extract_root}")
    extract_archive(archive_path, extract_root, remove_after_extract)


def download_url(
    url: str,
    rootpath: Path,
    filename: str,
    md5: str | None = None,
    max_redirect_hops: int = 3,
) -> Path:
    """Download a file from a URL and place it in root.

    Parameters
    ----------
    url : str
        URL to download file from
    rootpath : str
        Directory to place downloaded file in
    filename : str
        Name to save the file under.
    md5 : str, optional
        MD5 checksum of the download. If None, do not check.
    max_redirect_hops : int, optional
        Maximum number of redirect hops allowed.

    Returns
    -------
    pathlib.Path :
        Filepath to downloaded file.

    """
    rootpath = rootpath.expanduser()
    rootpath.mkdir(parents=True, exist_ok=True)
    filepath = rootpath / filename

    # check if file is already present locally
    if check_integrity(filepath, md5):
        print("Using already downloaded and verified file:", filepath)
        return filepath

    # expand redirect chain if needed
    url = get_redirect_url(url, max_hops=max_redirect_hops)

    # check if file is located on Google Drive
    #file_id = _get_google_drive_file_id(url)
    #if file_id is not None:
    #    return download_file_from_google_drive(file_id, root, filename, md5)

    # download the file
    try:
        print(f"Downloading {url} to {filepath}")
        retrieve_url(url, filepath)

    except (urllib.error.URLError, OSError) as e:  # type: ignore[attr-defined]
        if url[:5] == "https":
            url = url.replace("https:", "http:")
            print("Download failed. Trying https -> http instead.")
            print(f"Downloading {url} to {filepath}")
            retrieve_url(url, filepath)
        else:
            raise e

    # check integrity of downloaded file
    if not check_integrity(filepath, md5):
        raise RuntimeError(f"File {'filepath'} not found or corrupted.")

    return filepath


def check_integrity(
    filepath: Path,
    md5: str | None = None,
) -> bool:
    if not filepath.is_file():
        return False
    if md5 is None:
        return True
    return check_md5(filepath=filepath, md5=md5)


def check_md5(
    filepath: Path,
    md5: str,
    **kwargs: Any,
) -> bool:
    return md5 == calculate_md5(filepath, **kwargs)


def calculate_md5(
    filepath: Path,
    chunk_size: int = 1024 * 1024,
) -> str:
    # Setting the `usedforsecurity` flag does not change anything about the functionality, but
    # indicates that we are not using the MD5 checksum for cryptography. This enables its usage in
    # restricted environments like FIPS. Without it pymovements.datasets is unusable in these
    # environments since we perform a MD5 check everywhere.
    if sys.version_info >= (3, 9):
        md5 = hashlib.md5(usedforsecurity=False)
    else:
        md5 = hashlib.md5()

    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            md5.update(chunk)
    return md5.hexdigest()


def get_redirect_url(
    url: str,
    max_hops: int = 3,
) -> str:
    initial_url = url
    headers = {"Method": "HEAD", "User-Agent": USER_AGENT}

    for _ in range(max_hops + 1):
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers)) as response:
            if response.url == url or response.url is None:
                return url
            url = response.url

    else:
        raise RecursionError(
            f"Request to {initial_url} exceeded {max_hops} redirects."
            f" The last redirect points to {url}."
        )


def retrieve_url(
    url: str,
    filename: Path,
    chunk_size: int = 1024 * 32,
) -> None:
    header = {"User-Agent": USER_AGENT}
    with urllib.request.urlopen(urllib.request.Request(url, headers=header)) as response:
        save_response_content(
            content=iter(lambda: response.read(chunk_size), b""),
            destination=filename,
            length=response.length,
        )


def save_response_content(
    content: Iterator[bytes],
    destination: Path,
    length: int | None = None,
) -> None:
    with open(destination, "wb") as filehandler, tqdm(total=length) as pbar:
        for chunk in content:
            # filter out keep-alive chunks
            if not chunk:
                continue

            filehandler.write(chunk)
            pbar.update(len(chunk))


def extract_archive(
    source_path: Path,
    destination_path: Path | None = None,
    remove_after_extract: bool = False,
) -> Path:
    """Extract an archive.
    The archive type and a possible compression is automatically detected from the file name.
    If the file is compressed but not an archive the call is dispatched to :func:`_decompress`.

    Parameters
    ----------
    source_path : str
        Path to the file to be extracted.
    destination_path : str
        Path to the directory the file will be extracted to. If omitted, the directory of the file
         is used.
    remove_after_extract : bool
        If ``True``, remove the file after the extraction.

    Returns
    -------
    str :
        Path to the directory the file was extracted to.
    """
    if destination_path is None:
        destination_path = source_path.parent

    suffix, archive_type, compression = _detect_file_type(source_path)
    if not archive_type:
        return _decompress(
            source_path=source_path,
            destination_path=os.path.join(to_path, os.path.basename(from_path).replace(suffix, "")),
            remove_after_extract=remove_after_extract,
        )

    # We don't need to check for a missing key here, since this was already done in_detect_file_type()
    extractor = _ARCHIVE_EXTRACTORS[archive_type]

    extractor(source_path, destination_path, compression)
    if remove_after_extract:
        source_path.unlink()

    return destination_path


def _detect_file_type(filepath: Path) -> tuple[str, str | None, str | None]:
    """Detect the archive type and/or compression of a file.

    Parameters
    ----------
    filepath : Path
        The path of the file.

    Returns
    -------
    tuple :
        tuple of suffix, archive type, and compression

    Raises
    ------
        RuntimeError: if file has no suffix or suffix is not supported
    """
    suffixes = filepath.suffixes

    if not suffixes:
        raise RuntimeError(
            f"File '{filepath}' has no suffixes that could be used to detect the archive type and"
            " compression."
        )
    suffix = suffixes[-1]

    # check if the suffix is a known alias
    if suffix in _ARCHIVE_TYPE_ALIASES:
        return (suffix, *_ARCHIVE_TYPE_ALIASES[suffix])

    # check if the suffix is an archive type
    if suffix in _ARCHIVE_EXTRACTORS:
        return suffix, suffix, None

    # check if the suffix is a compression
    if suffix in _COMPRESSED_FILE_OPENERS:
        # check for suffix hierarchy
        if len(suffixes) > 1:
            suffix2 = suffixes[-2]

            # check if the suffix2 is an archive type
            if suffix2 in _ARCHIVE_EXTRACTORS:
                return suffix2 + suffix, suffix2, suffix

        return suffix, None, suffix

    valid_suffixes = sorted(set(_ARCHIVE_TYPE_ALIASES) | set(_ARCHIVE_EXTRACTORS) | set(_COMPRESSED_FILE_OPENERS))
    raise RuntimeError(f"Unknown compression or archive type: '{suffix}'.\nKnown suffixes are: '{valid_suffixes}'.")


import tarfile
import bz2
import gzip
import lzma
import zipfile


def _extract_tar(
    source_path: Path,
    destination_path: Path,
    compression: str | None,
) -> None:
    with tarfile.open(source_path, f"r:{compression[1:]}" if compression else "r") as tar_archive:
        tar_archive.extractall(destination_path)


_ZIP_COMPRESSION_MAP: dict[str, int] = {
    ".bz2": zipfile.ZIP_BZIP2,
    ".xz": zipfile.ZIP_LZMA,
}


def _extract_zip(
    source_path: Path,
    destination_path: Path,
    compression: str | None,
) -> None:
    with zipfile.ZipFile(
        source_path, "r",
        compression=_ZIP_COMPRESSION_MAP[compression] if compression else zipfile.ZIP_STORED
    ) as zip_archive:
        zip_archive.extractall(destination_path)


_ARCHIVE_EXTRACTORS: dict[str, Callable[[Path, Path, str | None], None]] = {
    ".tar": _extract_tar,
    ".zip": _extract_zip,
}

_ARCHIVE_TYPE_ALIASES: dict[str, tuple[str, str]] = {
    ".tbz": (".tar", ".bz2"),
    ".tbz2": (".tar", ".bz2"),
    ".tgz": (".tar", ".gz"),
}

_COMPRESSED_FILE_OPENERS: dict[str, Callable[..., IO]] = {
    ".bz2": bz2.open,
    ".gz": gzip.open,
    ".xz": lzma.open,
}


def _decompress(
    source_path: Path,
    destination_path: Path | None = None,
    remove_after_extract: bool = False,
) -> Path:
    r"""Decompress a file.

    The compression is automatically detected from the file name.

    Parameters
    ----------
    source_path : str
        Path to the file to be decompressed.
    destination_path : str
        Path to the decompressed file. If omitted, ``source_path`` without compression extension is
        used.
    remove_after_extract : bool
        If ``True``, remove the file after the extraction.

    Returns
    -------
    str : Path to the decompressed file.
    """
    suffix, archive_type, compression = _detect_file_type(source_path)
    if not compression:
        raise RuntimeError(f"Couldn't detect a compression from suffix {suffix}.")

    if destination_path is None:
        # TODO: this will actually rename the file
        # dont remove the assert before fixing this!!!!
        assert False
        destination_path = source_path.replace(suffix, archive_type if archive_type is not None else "")

    # We don't need to check for a missing key here, since this was already done in _detect_file_type()
    compressed_file_opener = _COMPRESSED_FILE_OPENERS[compression]

    with compressed_file_opener(from_path, "rb") as rfh, open(to_path, "wb") as wfh:
        wfh.write(rfh.read())

    if remove_after_extract:
        source_path.unlink()

    return destination_path
