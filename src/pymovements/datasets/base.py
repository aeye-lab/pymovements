from __future__ import annotations

import re
from pathlib import Path
from typing import Any
from urllib.error import URLError

import polars as pl
from tqdm.auto import tqdm

from pymovements.utils.downloads import download_and_extract_archive
from pymovements.utils.paths import get_filepaths


class Dataset:
    """

    """
    filename_regex = None
    filename_regex_dtypes = {}

    def __init__(
        self,
        root: str | Path,
        custom_csv_kwargs: dict[str, Any] | None = None,
    ):
        self.fileinfo = None
        self.data = None

        self.root = Path(root)

        if custom_csv_kwargs is None:
            custom_csv_kwargs = {}
        self.custom_csv_kwargs = custom_csv_kwargs

    def read(self):
        self.fileinfo = self.read_fileinfo()
        self.gaze = self.read_files()

    def read_fileinfo(self):
        if self.filename_regex is not None:
            filename_regex = re.compile(self.filename_regex)
        else:
            raise ValueError()

        # Get all filepaths that match regular expression.
        csv_filepaths = get_filepaths(
            rootpath=self.raw_dirpath,
            regex=filename_regex,
        )

        # Parse fileinfo from filenames.
        fileinfo_records = []
        for idx, filepath in enumerate(csv_filepaths):

            # All csv_filepaths already match the filename_regex.
            match = filename_regex.match(filepath.name)

            if match is None:
                raise RuntimeError(filepath)

            # We use the groupdict of the match as a base and add the filepath.
            fileinfo_record = match.groupdict()

            for fileinfo_key, fileinfo_dtype in self.filename_regex_dtypes.items():
                fileinfo_record[fileinfo_key] = fileinfo_dtype(fileinfo_record[fileinfo_key])

            fileinfo_record['filepath'] = str(filepath)
            fileinfo_records.append(fileinfo_record)

        # Create dataframe from all fileinfo records.
        fileinfo_df = pl.from_records(data=fileinfo_records)
        fileinfo_df = fileinfo_df.sort(by='filepath')

        return fileinfo_df

    def read_files(self):
        file_dfs = []

        # read and preprocess input files
        for file_id, filepath in enumerate(tqdm(self.fileinfo['filepath'])):
            file_df = pl.read_csv(filepath, **self.custom_csv_kwargs)

            for column in self.fileinfo.columns[::-1]:
                if column == 'filepath':
                    continue

                column_value = self.fileinfo.select(column)[file_id][0, 0]
                file_df = file_df.select([
                    pl.lit(column_value).alias(column),
                    pl.all()
                ])

            file_dfs.append(file_df)

        return file_dfs

    def compute_dva(self):
        for file_id, file_df in enumerate(tqdm(self.gaze)):
            pix_pos_cols = ['x_left_pixel', 'y_left_pixel', 'x_right_pixel', 'y_right_pixel']
            dva_pos_cols = ['x_left_dva', 'y_left_dva', 'x_right_dva', 'y_right_dva']

            pixel_data = file_df.select(pix_pos_cols)

            dva_data = self.experiment.screen.pix2deg(pixel_data.transpose(), center_origin=True)

            for dva_pos_col_id, dva_pos_col_name in enumerate(dva_pos_cols):
                self.gaze[file_id] = self.gaze[file_id].with_columns(
                    pl.Series(name=dva_pos_col_name, values=dva_data[:, dva_pos_col_id])
                )


    @property
    def dirpath(self) -> Path:
        return self.root / self.__class__.__name__.lower()

    @property
    def raw_dirpath(self) -> Path:
        return self.dirpath / "raw"


class PublicDataset(Dataset):
    # TODO: add abstractmethod decorator
    mirrors = None
    resources = None

    def __init__(
        self,
        root: str,
        download: bool = False,
        remove_finished: bool = False,
        **kwargs,
    ):
        # FIXME: This is ugly as we're redoing this in super().__init__()
        self.root = Path(root)

        if download:
            self.download(remove_finished=remove_finished)

        super().__init__(root=root, **kwargs)

    def download(self, remove_finished: bool = False):
        if not self.mirrors:
            raise ValueError("no mirrors defined for dataset")

        if not self.resources:
            raise ValueError("no resources defined for datasaet")

        self.raw_dirpath.mkdir(parents=True, exist_ok=True)

        for resource in self.resources:
            for mirror in self.mirrors:

                url = f'{mirror}{resource["path"]}'

                try:
                    download_and_extract_archive(
                        url=url,
                        download_dirpath=self.dirpath,
                        download_filename=resource['filename'],
                        extract_dirpath=self.raw_dirpath,
                        md5=resource['md5'],
                        remove_finished=remove_finished,
                    )

                except URLError as error:
                    print(f"Failed to download (trying next):\n{error}")
                    # downloading the resource, try next mirror
                    continue
                    # TODO: check that at least one mirror was successful

                # downloading the resource was successful, we don't need to try another mirror
                break
