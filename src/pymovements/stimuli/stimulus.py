# Copyright (c) 2022-2024 The pymovements Project Authors
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
"""Provides the StimulusDataFrame class."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl
from tqdm.auto import tqdm

from pymovements.stimuli import stimulus_files


class StimulusDataFrame:
    """StimulusDataFrame base class.

    Initialize the StimulusDataFrame object.

    path : str | Path | DatasetPaths
        Path to the dataset directory. You can set up a custom directory structure by passing a
        :py:class:`~pymovements.DatasetPaths` instance.
    """

    def __init__(
            self,
    ):
        self.fileinfo: pl.DataFrame = pl.DataFrame()
        self.stimulus: list[StimulusDataFrame] = []

    def scan(self) -> Dataset:
        """Infer information from filepaths and filenames.

        Returns
        -------
        Dataset
            Returns self, useful for method cascading.

        Raises
        ------
        AttributeError
            If no regular expression for parsing filenames is defined.
        RuntimeError
            If an error occurred during matching filenames or no files have been found.
        """
        self.fileinfo = dataset_files.scan_dataset(definition=self.definition, paths=self.paths)
        return self

    def load_file(
            self,
            stimulus_filepath: str | Path,
            custom_read_kwargs: dict[str, Any] | None = None,
    ) -> Dataset:
        """Load all available stimulus data files.

        Parameters
        ----------
        preprocessed: bool
            If ``True``, saved preprocessed data will be loaded, otherwise raw data will be loaded.
            (default: False)
        preprocessed_dirname: str | None
            One-time usage of an alternative directory name to save data relative to
            :py:meth:`pymovements.Dataset.path`.
            This argument is used only for this single call and does not alter
            :py:meth:`pymovements.Dataset.preprocessed_rootpath`. (default: None)
        extension: str
            Specifies the file format for loading data. Valid options are: `csv`, `feather`.
            (default: 'feather')

        Returns
        -------
        Dataset
            Returns self, useful for method cascading.

        Raises
        ------
        AttributeError
            If `fileinfo` is None or the `fileinfo` dataframe is empty.
        RuntimeError
            If file type of stimulus file is not supported.
        """
        if isinstance(stimulus_filepath, str):
            stimulus_filepath = Path(stimulus_filepath)
        self.stimulus = stimulus_files.load_stimulus_file(
            stimulus_filepath,
            custom_read_kwargs,
        )
        return self

    def save(
            self,
            dirname: str | None = None,
            verbose: int = 1,
            extension: str = 'feather',
    ) -> Dataset:
        """Save preprocessed stimulus and event files.

        Data will be saved as feather/csv files to ``Dataset.preprocessed_roothpath`` or
        ``Dataset.events_roothpath`` with the same directory structure as the raw data.

        Returns
        -------
        Dataset
            Returns self, useful for method cascading.

        Parameters
        ----------
        events_dirname: str | None
            One-time usage of an alternative directory name to save data relative to dataset path.
            This argument is used only for this single call and does not alter
            :py:meth:`pymovements.Dataset.events_rootpath`. (default: None)
        preprocessed_dirname: str | None
            One-time usage of an alternative directory name to save data relative to dataset path.
            This argument is used only for this single call and does not alter
            :py:meth:`pymovements.Dataset.preprocessed_rootpath`. (default: None)
        verbose: int
            Verbosity level (0: no print output, 1: show progress bar, 2: print saved filepaths)
            (default: 1)
        extension: str
            Extension specifies the fileformat to store the data. (default: 'feather')
        """
        self.save_events(dirname, verbose=verbose, extension=extension)
        return self
