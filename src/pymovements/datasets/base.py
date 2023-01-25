from __future__ import annotations

from pathlib import Path
from urllib.error import URLError

from pymovements.utils.downloads import download_and_extract_archive


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

