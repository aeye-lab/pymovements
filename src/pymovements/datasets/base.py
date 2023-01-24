from __future__ import annotations
from pathlib import Path
from urllib.error import URLError
from typing import Any
import hashlib
import sys
import urllib.request


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

                url = f'{mirror}/{resource["path"]}'

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
    filepath : pathlib.Path
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
        _urlretrieve(url, filepath)
    except (urllib.error.URLError, OSError) as e:  # type: ignore[attr-defined]
        if url[:5] == "https":
            url = url.replace("https:", "http:")
            print("Failed download. Trying https -> http instead. Downloading " + url + " to " + filepath)
            _urlretrieve(url, filepath)
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
    filepath: str,
    md5: str,
    **kwargs: Any,
) -> bool:
    return md5 == calculate_md5(filepath, **kwargs)


def calculate_md5(
    filepath: str,
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


def get_redirect_url(url: str, max_hops: int = 3) -> str:
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


def _urlretrieve(url: str, filename: str, chunk_size: int = 1024 * 32) -> None:
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": USER_AGENT})) as response:
        _save_response_content(iter(lambda: response.read(chunk_size), b""), filename, length=response.length)


from tqdm import tqdm
from typing import Optional, Iterator

def _save_response_content(
    content: Iterator[bytes],
    destination: str,
    length: Optional[int] = None,
) -> None:
    with open(destination, "wb") as fh, tqdm(total=length) as pbar:
        for chunk in content:
            # filter out keep-alive new chunks
            if not chunk:
                continue

            fh.write(chunk)
            pbar.update(len(chunk))
