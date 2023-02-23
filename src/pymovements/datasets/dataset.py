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
"""This module provides the base dataset class."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import polars as pl
from tqdm.auto import tqdm

from pymovements.base import Experiment
from pymovements.events.events import Event
from pymovements.events.events import EventDetectionCallable
from pymovements.utils.paths import get_filepaths


class Dataset:
    """Dataset base class."""

    def __init__(
        self,
        root: str | Path,
        experiment: Experiment | None = None,
        filename_regex: str = '.*',
        filename_regex_dtypes: dict[str, type] | None = None,
        custom_read_kwargs: dict[str, Any] | None = None,
    ):
        """Initialize the dataset object.

        Parameters
        ----------
        root : str, Path
            Path to the root directory of the dataset.
        experiment : Experiment
            The experiment definition.
        filename_regex : str
            Regular expression which needs to be matched before trying to load the file. Named
            groups will appear in the `fileinfo` dataframe.
        filename_regex_dtypes : dict[str, type], optional
            If named groups are present in the `filename_regex`, this makes it possible to cast
            specific named groups to a particular datatype.
        custom_read_kwargs : dict[str, Any], optional
            If specified, these keyword arguments will be passed to the file reading function.
        """
        self.fileinfo: pl.DataFrame = pl.DataFrame()
        self.gaze: list[pl.DataFrame] = []
        self.events: list[pl.DataFrame] = []

        self.root = Path(root)

        self.experiment = experiment

        self._filename_regex = filename_regex

        if filename_regex_dtypes is None:
            filename_regex_dtypes = {}
        self._filename_regex_dtypes = filename_regex_dtypes

        if custom_read_kwargs is None:
            custom_read_kwargs = {}
        self._custom_read_kwargs = custom_read_kwargs

    def load(
            self,
            events: bool = False,
            preprocessed: bool = False,
            subset: None | dict[str, float | int | str | list[float | int | str]] = None,
    ):
        """Parse file information and load all gaze files.

        Parameters
        ----------
        events : bool
            If ``True``, load previously saved event data.
        preprocessed : bool
            If ``True``, load previously saved preprocessed data, otherwise load raw data.
        subset : dict, optional
            If specified, load only a subset of the dataset. All keys in the dictionary must be
            present in the fileinfo dataframe inferred by `infer_fileinfo()`. Values can be either
            float, int , str or a list of these.

        The parsed file information is assigned to the `fileinfo` attribute.
        All gaze files will be loaded as dataframes and assigned to the `gaze` attribute.
        """
        self.fileinfo = self.infer_fileinfo()
        self.take_subset(subset=subset)
        self.gaze = self.load_gaze_files(preprocessed=preprocessed)

        if events:
            self.events = self.load_event_files()

    def infer_fileinfo(self) -> pl.DataFrame:
        """Infer information from filepaths and filenames.

        Returns
        -------
        pl.DataFrame :
            File information dataframe.

        Raises
        ------
        AttributeError
            If no regular expression for parsing filenames is defined.
        RuntimeError
            If an error occured during matching filenames or no files have been found.
        """
        if self._filename_regex is not None:
            filename_regex = re.compile(self._filename_regex)
        else:
            raise AttributeError('no regular expression for filenames is defined.')

        # Get all filepaths that match regular expression.
        csv_filepaths = get_filepaths(
            path=self.raw_rootpath,
            regex=filename_regex,
        )

        # Parse fileinfo from filenames.
        fileinfo_dicts: list[dict[str, Any]] = []
        for filepath in csv_filepaths:

            # All csv_filepaths already match the filename_regex.
            match = filename_regex.match(filepath.name)

            # This actually should never happen but mypy will complain otherwise.
            if match is None:
                raise RuntimeError(
                    f'file {filepath} did not match regular expression {self._filename_regex}',
                )

            # We use the groupdict of the match as a base and add the filepath.
            fileinfo_dict = match.groupdict()

            for fileinfo_key, fileinfo_dtype in self._filename_regex_dtypes.items():
                fileinfo_dict[fileinfo_key] = fileinfo_dtype(fileinfo_dict[fileinfo_key])

            fileinfo_dict['filepath'] = str(filepath)
            fileinfo_dicts.append(fileinfo_dict)

        if len(fileinfo_dicts) == 0:
            raise RuntimeError('no matching files found')

        # Create dataframe from all fileinfo records.
        fileinfo_df = pl.from_dicts(dicts=fileinfo_dicts, infer_schema_length=1)
        fileinfo_df = fileinfo_df.sort(by='filepath')

        return fileinfo_df

    def take_subset(
            self,
            subset: None | dict[str, float | int | str | list[float | int | str]] = None,
    ) -> None:
        """Take a subset of the dataset.

        Calling this method will alter the fileinfo attribute.

        Parameters
        ----------
        subset : dict, optional
            If specified, load only a subset of the dataset. All keys in the dictionary must be
            present in the fileinfo dataframe inferred by `infer_fileinfo()`. Values can be either
            float, int , str or a list of these.

        Raises
        ------
        ValueError
            If dictionary key is not a column in the fileinfo dataframe.
        TypeError
            If dictionary key or value is not of valid type.

        """
        if subset is None:
            return

        for subset_key, subset_value in subset.items():
            if subset_key not in self.fileinfo.columns:
                raise ValueError(
                    f'subset key {subset_key} must be a column in the fileinfo attribute.'
                    f' Available columns are: {self.fileinfo.columns}',
                )

            if not isinstance(subset_key, str):
                raise TypeError(
                    f'subset keys must be of type str but key {subset_key} is of type'
                    f' {type(subset_key)}',
                )

            if isinstance(subset_value, (float, int, str)):
                column_values = [subset_value]
            elif isinstance(subset_value, (list, tuple)):
                column_values = subset_value
            else:
                raise TypeError(
                    f'subset value must be of float, int, str or a list of these but key-value pair'
                    f'{subset_key}: {subset_value} is of type {type(subset_value)}',
                )

            self.fileinfo = self.fileinfo.filter(pl.col(subset_key).is_in(column_values))

    def load_gaze_files(self, preprocessed: bool = False) -> list[pl.DataFrame]:
        """Load all available gaze data files.

        Parameters
        ----------
        preprocessed : bool
            If ``True``, saved preprocessed data will be loaded, otherwise raw data will be loaded.

        Returns
        -------
        list[pl.DataFrame]
            List of gaze dataframes.

        Raises
        ------
        AttributeError
            If `fileinfo` is None or the `fileinfo` dataframe is empty.
        RuntimeError
            If file type of gaze file is not supported.
        """
        self._check_fileinfo()

        file_dfs: list[pl.DataFrame] = []

        # Read gaze files from fileinfo attribute.
        for file_id, filepath in enumerate(tqdm(self.fileinfo['filepath'])):
            filepath = Path(filepath)

            if preprocessed:
                filepath = self._raw_to_preprocessed_filepath(filepath)

            if filepath.suffix == '.csv':
                file_df = pl.read_csv(filepath, **self._custom_read_kwargs)
            elif filepath.suffix == '.feather':
                file_df = pl.read_ipc(filepath)
            else:
                raise RuntimeError('data files of type {filepath.suffix} are not supported.')

            # Add fileinfo columns to dataframe.
            for column in self.fileinfo.columns[::-1]:
                # Preprocessed data already has fileinfo columns.
                if preprocessed:
                    continue

                if column == 'filepath':
                    continue

                column_value = self.fileinfo.select(column)[file_id][0, 0]
                file_df = file_df.select([
                    pl.lit(column_value).alias(column),
                    pl.all(),
                ])

            file_dfs.append(file_df)

        return file_dfs

    def load_event_files(self) -> list[pl.DataFrame]:
        """Load all available event files.

        Returns
        -------
        list[pl.DataFrame]
            List of event dataframes.

        Raises
        ------
        AttributeError
            If `fileinfo` is None or the `fileinfo` dataframe is empty.
        """
        self._check_fileinfo()

        file_dfs: list[pl.DataFrame] = []

        # read and preprocess input files
        for _, filepath in enumerate(tqdm(self.fileinfo['filepath'])):
            filepath = Path(filepath)

            filepath = self._raw_to_event_filepath(filepath)

            file_df = pl.read_ipc(filepath)

            file_dfs.append(file_df)

        return file_dfs

    def pix2deg(self, verbose: bool = True) -> None:
        """Compute gaze positions in degrees of visual angle from pixel coordinates.

        This requires an experiment definition and also assumes that the columns 'x_left_pix',
         'y_left_pix', 'x_right_pix' and 'y_right_pix' are available in the gaze dataframe.

        After success, the gaze dataframe is extended by the columns 'x_left_dva', 'y_left_dva',
        'x_right_dva' and, 'y_right_dva'.

        Parameters
        ----------
        verbose : bool
            If True, show progress of computation.

        Raises
        ------
        AttributeError
            If `gaze` is None or there are no gaze dataframes present in the `gaze` attribute, or
            if experiment is None.
        """
        if self.gaze is None:
            raise AttributeError(
                'gaze files were not loaded yet. please run load() or load_gaze_files() beforehand',
            )
        if len(self.gaze) == 0:
            raise AttributeError('no files present in gaze attribute')
        if self.experiment is None:
            raise AttributeError('experiment must be specified for this method.')

        disable_progressbar = not verbose

        for file_id, file_df in enumerate(tqdm(self.gaze, disable=disable_progressbar)):
            pix_position_columns = ['x_left_pix', 'y_left_pix', 'x_right_pix', 'y_right_pix']
            dva_position_columns = ['x_left_dva', 'y_left_dva', 'x_right_dva', 'y_right_dva']

            pixel_positions = file_df.select(pix_position_columns)

            dva_positions = self.experiment.screen.pix2deg(pixel_positions.to_numpy())

            for dva_column_id, dva_column_name in enumerate(dva_position_columns):
                self.gaze[file_id] = self.gaze[file_id].with_columns(
                    pl.Series(name=dva_column_name, values=dva_positions[:, dva_column_id]),
                )

    def pos2vel(self, method: str = 'smooth', verbose: bool = True, **kwargs) -> None:
        """Compute gaze velocites in dva/s from dva coordinates.

        This requires an experiment definition and also assumes that the columns 'x_left_dva',
         'y_left_dva', 'x_right_dva' and 'y_right_dva' are available in the gaze dataframe.

        After success, the gaze dataframe is extended by the columns 'x_left_vel', 'y_left_vel',
        'x_right_vel' and, 'y_right_vel'.

        Parameters
        ----------
        method : str
            Computation method. See :func:`~transforms.pos2vel()` for details, default: smooth.
        verbose : bool
            If True, show progress of computation.
        **kwargs
            Additional keyword arguments to be passed to the :func:`~transforms.pos2vel()` method.

        Raises
        ------
        AttributeError
            If `gaze` is None or there are no gaze dataframes present in the `gaze` attribute, or
            if experiment is None.
        """
        if self.gaze is None:
            raise AttributeError(
                'gaze files were not loaded yet. please run load() or load_gaze_files() beforehand',
            )
        if len(self.gaze) == 0:
            raise AttributeError('no files present in gaze attribute')
        if self.experiment is None:
            raise AttributeError('experiment must be specified for this method.')

        disable_progressbar = not verbose

        for file_id, file_df in enumerate(tqdm(self.gaze, disable=disable_progressbar)):
            position_columns = ['x_left_dva', 'y_left_dva', 'x_right_dva', 'y_right_dva']
            velocity_columns = ['x_left_vel', 'y_left_vel', 'x_right_vel', 'y_right_vel']

            positions = file_df.select(position_columns)

            velocities = self.experiment.pos2vel(positions.to_numpy(), method=method, **kwargs)

            for col_id, velocity_column_name in enumerate(velocity_columns):
                self.gaze[file_id] = self.gaze[file_id].with_columns(
                    pl.Series(name=velocity_column_name, values=velocities[:, col_id]),
                )

    def detect_events(
            self,
            method: EventDetectionCallable,
            eye: str = 'auto',
            clear: bool = False,
            verbose: bool = True,
            **kwargs,
    ) -> None:
        """Detect events by applying a specific event detection method.

        Parameters
        ----------
        method : EventDetectionCallable
            The event detection method to be applied.
        eye : str
            Select which eye to choose. Valid options are ``auto``, ``left`` or ``right``. If
            ``auto`` is passed, ``left`` will only be chosen if the right eye is not available in
            the gaze data frame.
        clear : bool
            If ``True``, event DataFrame will be overwritten with new DataFrame instead of being
             merged into the existing one.
        verbose : bool
            If ``True``, show progress bar.
        **kwargs :
            Additional keyword arguments to be passed to the event detection method.

        """
        if self.gaze is None:
            raise AttributeError(
                'gaze files were not loaded yet. please run load() or load_gaze_files() beforehand',
            )
        if len(self.gaze) == 0:
            raise AttributeError('no files present in gaze attribute')

        # Automatically infer eye to use for event detection.
        if eye == 'auto':
            if 'x_right_dva' in self.gaze[0].columns:
                eye = 'right'
            elif 'x_left_dva' in self.gaze[0].columns:
                eye = 'left'
            else:
                raise AttributeError(
                    'Either right or left eye columns must be present in gaze data frame.'
                    f' Available columns are: {self.gaze[0].columns}',
                )

        position_columns = [f'x_{eye}_dva', f'y_{eye}_dva']
        velocity_columns = [f'x_{eye}_vel', f'y_{eye}_vel']

        disable_progressbar = not verbose

        event_dfs: list[pl.DataFrame] = []

        for gaze_df in tqdm(self.gaze, disable=disable_progressbar):

            positions = gaze_df.select(position_columns).to_numpy()
            velocities = gaze_df.select(velocity_columns).to_numpy()

            events = method(positions=positions, velocities=velocities, **kwargs)
            event_dfs.append(events)

        if not self.events or clear:
            self.events = event_dfs
            return

        for file_id, event_df in enumerate(event_dfs):
            self.events[file_id] = pl.concat(
                [self.events[file_id], event_df],
                how='diagonal',
            )

    def clear_events(self) -> None:
        """Clear event DataFrame."""
        if len(self.events) == 0:
            return

        for file_id, _ in enumerate(self.events):
            self.events[file_id] = pl.DataFrame(schema=Event.schema)

    def save(self, verbose: int = 1):
        """Save preprocessed gaze and event files.

        Data will be saved as feather files to ``Dataset.preprocessed_roothpath`` or
        ``Dataset.events_roothpath`` with the same directory structure as the raw data.

        Parameters
        ----------
        verbose : int
            Verbosity level (0: no print output, 1: show progress bar, 2: print saved filepaths)
        """
        if verbose > 2:
            print('save preprocessed gaze files')
        self.save_preprocessed(verbose=verbose)

        if verbose > 2:
            print('save event files')
        self.save_events(verbose=verbose)

    def save_events(self, verbose: int = 1):
        """Save events to files.

        Data will be saved as feather files to ``Dataset.events_roothpath`` with the same directory
        structure as the raw data.

        Parameters
        ----------
        verbose : int
            Verbosity level (0: no print output, 1: show progress bar, 2: print saved filepaths)
        """
        disable_progressbar = not verbose

        for file_id, event_df in enumerate(tqdm(self.events, disable=disable_progressbar)):
            raw_filepath = Path(self.fileinfo[file_id, 'filepath'])
            events_filepath = self._raw_to_event_filepath(raw_filepath)

            events_filepath.parent.mkdir(parents=True, exist_ok=True)

            if verbose > 2:
                print('Save file to', events_filepath)

            event_df.write_ipc(events_filepath)

    def save_preprocessed(self, verbose: int = 1):
        """Save preprocessed gaze files.

        Data will be saved as feather files to ``Dataset.preprocessed_roothpath`` with the same
        directory structure as the raw data.

        Parameters
        ----------
        verbose : int
            Verbosity level (0: no print output, 1: show progress bar, 2: print saved filepaths)
        """
        disable_progressbar = not verbose

        for file_id, gaze_df in enumerate(tqdm(self.gaze, disable=disable_progressbar)):
            raw_filepath = Path(self.fileinfo[file_id, 'filepath'])
            preprocessed_filepath = self._raw_to_preprocessed_filepath(raw_filepath)

            preprocessed_filepath.parent.mkdir(parents=True, exist_ok=True)

            if verbose > 2:
                print('Save file to', preprocessed_filepath)

            gaze_df.write_ipc(preprocessed_filepath)

    @property
    def rootpath(self) -> Path:
        """Get the path to the dataset directory.

        The dataset path points to a directory in the specified root directory which is named the
        same as the respective class.

        Example
        -------
        >>> class CustomDataset(Dataset):
        ...     pass
        >>> dataset = CustomDataset(root='data')
        >>> dataset.rootpath  # doctest: +SKIP
        Path('data/CustomDataset')
        """
        return self.root / self.__class__.__name__

    @property
    def events_rootpath(self) -> Path:
        """Get the path to the directory of the event data.

        The event data directory path points to a directory named `events` in the dataset
        `rootpath`.

        Example
        -------
        >>> class CustomDataset(Dataset):
        ...     pass
        >>> dataset = CustomDataset(root='data')
        >>> dataset.events_rootpath  # doctest: +SKIP
        Path('data/CustomDataset/events')
        """
        return self.rootpath / 'events'

    @property
    def preprocessed_rootpath(self) -> Path:
        """Get the path to the directory of the preprocessed gaze data.

        The preprocessed data directory path points to a directory named `preprocessed` in the
        dataset `rootpath`.

        Example
        -------
        >>> class CustomDataset(Dataset):
        ...     pass
        >>> dataset = CustomDataset(root='data')
        >>> dataset.preprocessed_rootpath  # doctest: +SKIP
        Path('data/CustomDataset/preprocessed')
        """
        return self.rootpath / 'preprocessed'

    @property
    def raw_rootpath(self) -> Path:
        """Get the path to the directory of the raw data.

        The raw data directory path points to a directory named `raw` in the dataset `rootpath`.

        Example
        -------
        >>> class CustomDataset(Dataset):
        ...     pass
        >>> dataset = CustomDataset(root='data')
        >>> dataset.raw_rootpath  # doctest: +SKIP
        Path('data/CustomDataset/raw')
        """
        return self.rootpath / 'raw'

    def _check_fileinfo(self) -> None:
        """Check if fileinfo attribute is set and there is at least one row present."""
        if self.fileinfo is None:
            raise AttributeError(
                'fileinfo was not loaded yet. please run load() or infer_fileinfo() beforehand',
            )
        if len(self.fileinfo) == 0:
            raise AttributeError('no files present in fileinfo attribute')

    def _raw_to_preprocessed_filepath(self, raw_filepath: Path) -> Path:
        """Get preprocessed filepath in accordance to filepath of the raw file.

        The preprocessed filepath will point to a feather file.

        Parameters
        ----------
        raw_filepath : Path
            The Path to the raw file.

        Returns
        -------
        Path
            The Path to the preprocessed feather file.
        """
        relative_raw_dirpath = raw_filepath.parent.relative_to(self.raw_rootpath)
        preprocessed_file_dirpath = self.preprocessed_rootpath / relative_raw_dirpath

        # Get new filename for saved feather file.
        preprocessed_filename = raw_filepath.stem + '.feather'

        return preprocessed_file_dirpath / preprocessed_filename

    def _raw_to_event_filepath(self, raw_filepath: Path) -> Path:
        """Get event filepath in accordance to filepath of the raw file.

        The event filepath will point to a feather file.

        Parameters
        ----------
        raw_filepath : Path
            The Path to the raw file.

        Returns
        -------
        Path
            The Path to the event feather file.
        """
        relative_raw_dirpath = raw_filepath.parent.relative_to(self.raw_rootpath)
        events_file_dirpath = self.events_rootpath / relative_raw_dirpath

        # Get new filename for saved feather file.
        events_filename = raw_filepath.stem + '.feather'

        return events_file_dirpath / events_filename
