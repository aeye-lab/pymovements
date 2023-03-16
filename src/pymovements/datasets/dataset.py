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

from pymovements.events.event_processing import EventGazeProcessor
from pymovements.events.events import EventDataFrame
from pymovements.events.events import EventDetectionCallable
from pymovements.gaze import GazeDataFrame
from pymovements.gaze.experiment import Experiment
from pymovements.utils.paths import match_filepaths


class Dataset:
    """Dataset base class."""
    # pylint: disable=too-many-instance-attributes
    # The Dataset class is exceptionally complex and needs many attributes.

    def __init__(
            self,
            root: str | Path,
            experiment: Experiment | None = None,
            filename_regex: str = '.*',
            filename_regex_dtypes: dict[str, type] | None = None,
            custom_read_kwargs: dict[str, Any] | None = None,
            dataset_dirname: str = '.',
            raw_dirname: str = 'raw',
            preprocessed_dirname: str = 'preprocessed',
            events_dirname: str = 'events',
    ):
        """Initialize the dataset object.

        You can set up a custom directory structure by populating the particular dirname attributes.
        See :py:attr:`~pymovements.dataset.Dataset.dataset_dirname`,
        :py:attr:`~pymovements.dataset.Dataset.raw_dirname`,
        :py:attr:`~pymovements.dataset.Dataset.preprocessed_dirname` and
        :py:attr:`~pymovements.dataset.Dataset.events_dirname` for details.

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
        dataset_dirname : str, optional
            Dataset directory name under root path. Can be `.` if dataset is located in root path.
            Default: `.`
        raw_dirname ; str, optional
            Name of directory under dataset path that contains raw data. Can be `.` if raw data is
            located in dataset path. We advise the user to keep the original raw data separate from
            the preprocessed / event data. Default: `raw`
        preprocessed_dirname : str, optional
            Name of directory under dataset path that will be used to store preprocessed data. We
            advise the user to keep the preprocessed data separate from the original raw data.
            Default: `preprocessed`
        events_dirname : str, optional
            Name of directory under dataset path that will be used to store event data. We advise
            the user to keep the event data separate from the original raw data. Default: `events`
        """
        self.fileinfo: pl.DataFrame = pl.DataFrame()
        self.gaze: list[GazeDataFrame] = []
        self.events: list[EventDataFrame] = []

        self._root = Path(root)
        self.dataset_dirname = dataset_dirname
        self.raw_dirname = raw_dirname
        self.preprocessed_dirname = preprocessed_dirname
        self.events_dirname = events_dirname

        self.experiment = experiment

        if filename_regex is None:
            raise ValueError('filename_regex must not be None')
        if not isinstance(filename_regex, str):
            raise TypeError('filename_regex must be of type str')
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
            events_dirname: str | None = None,
            preprocessed_dirname: str | None = None,
            extension: str = 'feather',
    ) -> Dataset:
        """Parse file information and load all gaze files.

        The parsed file information is assigned to the `fileinfo` attribute.
        All gaze files will be loaded as dataframes and assigned to the `gaze` attribute.

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
        events_dirname : str
            One-time usage of an alternative directory name to save data relative to
            :py:meth:`pymovements.Dataset.path`.
            This argument is used only for this single call and does not alter
            :py:meth:`pymovements.Dataset.events_rootpath`.
        preprocessed_dirname : str
            One-time usage of an alternative directory name to save data relative to
            :py:meth:`pymovements.Dataset.path`.
            This argument is used only for this single call and does not alter
            :py:meth:`pymovements.Dataset.preprocessed_rootpath`.
        extension:
            Specifies the file format for loading data. Valid options are: `csv`, `feather`.
            :Default: `feather`.

        Returns
        -------
        Dataset
            Returns self, useful for method cascading.
        """
        fileinfo = self.infer_fileinfo()
        self.fileinfo = self.take_subset(fileinfo=fileinfo, subset=subset)
        self.gaze = self.load_gaze_files(
            preprocessed=preprocessed, preprocessed_dirname=preprocessed_dirname,
            extension=extension,
        )

        if events:
            self.events = self.load_event_files(
                events_dirname=events_dirname,
                extension=extension,
            )

        return self

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
            If an error occurred during matching filenames or no files have been found.
        """
        # Get all filepaths that match regular expression.
        fileinfo_dicts = match_filepaths(
            path=self.raw_rootpath,
            regex=re.compile(self._filename_regex),
            relative=True,
        )

        if len(fileinfo_dicts) == 0:
            raise RuntimeError(f'no matching files found in {self.raw_rootpath}')

        # Create dataframe from all fileinfo records.
        fileinfo_df = pl.from_dicts(data=fileinfo_dicts, infer_schema_length=1)
        fileinfo_df = fileinfo_df.sort(by='filepath')

        fileinfo_df = fileinfo_df.with_columns([
            pl.col(fileinfo_key).cast(fileinfo_dtype)
            for fileinfo_key, fileinfo_dtype in self._filename_regex_dtypes.items()
        ])

        return fileinfo_df

    @staticmethod
    def take_subset(
            fileinfo: pl.DataFrame,
            subset: None | dict[
                str, bool | float | int | str | list[bool | float | int | str],
            ] = None,
    ) -> pl.DataFrame:
        """Take a subset of the dataset.

        Calling this method will alter the fileinfo attribute.

        Parameters
        ----------
        fileinfo : pl.DataFrame
            File information dataframe.
        subset : dict, optional
            If specified, take a subset of the dataset. All keys in the dictionary must be
            present in the fileinfo dataframe inferred by `infer_fileinfo()`. Values can be either
            bool, float, int , str or a list of these.

        Returns
        -------
        pl.DataFrame:
            Subset of file information dataframe.

        Raises
        ------
        ValueError
            If dictionary key is not a column in the fileinfo dataframe.
        TypeError
            If dictionary key or value is not of valid type.

        """
        if subset is None:
            return fileinfo

        if not isinstance(subset, dict):
            raise TypeError(f'subset must be of type dict but is of type {type(subset)}')

        for subset_key, subset_value in subset.items():
            if not isinstance(subset_key, str):
                raise TypeError(
                    f'subset keys must be of type str but key {subset_key} is of type'
                    f' {type(subset_key)}',
                )

            if subset_key not in fileinfo.columns:
                raise ValueError(
                    f'subset key {subset_key} must be a column in the fileinfo attribute.'
                    f' Available columns are: {fileinfo.columns}',
                )

            if isinstance(subset_value, (bool, float, int, str)):
                column_values = [subset_value]
            elif isinstance(subset_value, (list, tuple)):
                column_values = subset_value
            else:
                raise TypeError(
                    f'subset value must be of type bool, float, int, str or a list of these but'
                    f' key-value pair {subset_key}: {subset_value} is of type {type(subset_value)}',
                )

            fileinfo = fileinfo.filter(pl.col(subset_key).is_in(column_values))
        return fileinfo

    def load_gaze_files(
            self,
            preprocessed: bool = False,
            preprocessed_dirname: str | None = None,
            extension: str = 'feather',
    ) -> list[GazeDataFrame]:
        """Load all available gaze data files.

        Parameters
        ----------
        preprocessed : bool
            If ``True``, saved preprocessed data will be loaded, otherwise raw data will be loaded.
        preprocessed_dirname : str
            One-time usage of an alternative directory name to save data relative to
            :py:meth:`pymovements.Dataset.path`.
            This argument is used only for this single call and does not alter
            :py:meth:`pymovements.Dataset.preprocessed_rootpath`.
        extension:
            Specifies the file format for loading data. Valid options are: `csv`, `feather`.
            :Default: `feather`.

        Returns
        -------
        list[GazeDataFrame]
            List of gaze dataframes.

        Raises
        ------
        AttributeError
            If `fileinfo` is None or the `fileinfo` dataframe is empty.
        RuntimeError
            If file type of gaze file is not supported.
        """
        self._check_fileinfo()

        gaze_dfs: list[GazeDataFrame] = []

        # Read gaze files from fileinfo attribute.
        for fileinfo in tqdm(self.fileinfo.to_dicts()):
            filepath = Path(fileinfo['filepath'])
            filepath = self.raw_rootpath / filepath

            if preprocessed:
                filepath = self._raw_to_preprocessed_filepath(
                    filepath, preprocessed_dirname=preprocessed_dirname,
                    extension=extension,
                )

            if filepath.suffix == '.csv':
                if preprocessed:
                    gaze_df = pl.read_csv(filepath)
                else:
                    gaze_df = pl.read_csv(filepath, **self._custom_read_kwargs)
            elif filepath.suffix == '.feather':
                gaze_df = pl.read_ipc(filepath)
            else:
                raise RuntimeError(f'data files of type {filepath.suffix} are not supported')

            # Add fileinfo columns to dataframe.
            gaze_df = self._add_fileinfo(gaze_df, fileinfo)

            gaze_dfs.append(GazeDataFrame(gaze_df, experiment=self.experiment))

        return gaze_dfs

    def load_event_files(
        self,
        events_dirname: str | None = None,
        extension: str = 'feather',
    ) -> list[EventDataFrame]:
        """Load all available event files.

        Parameters
        ----------
        events_dirname : str
            One-time usage of an alternative directory name to save data relative to
            :py:meth:`pymovements.Dataset.path`.
            This argument is used only for this single call and does not alter
            :py:meth:`pymovements.Dataset.events_rootpath`.
        extension:
            Specifies the file format for loading data. Valid options are: `csv`, `feather`.
            :Default: `feather`.

        Returns
        -------
        list[EventDataFrame]
            List of event dataframes.

        Raises
        ------
        AttributeError
            If `fileinfo` is None or the `fileinfo` dataframe is empty.
        ValueError
            If extension is not in list of valid extensions.
        """
        self._check_fileinfo()

        event_dfs: list[EventDataFrame] = []

        # read and preprocess input files
        for fileinfo in tqdm(self.fileinfo.to_dicts()):
            filepath = Path(fileinfo['filepath'])
            filepath = self.raw_rootpath / filepath

            filepath = self._raw_to_event_filepath(
                filepath, events_dirname=events_dirname,
                extension=extension,
            )

            if extension == 'feather':
                event_df = pl.read_ipc(filepath)
            elif extension == 'csv':
                event_df = pl.read_csv(filepath)
            else:
                valid_extensions = ['csv', 'feather']
                raise ValueError(
                    f'unsupported file format "{extension}".'
                    f'Supported formats are: {valid_extensions}',
                )

            # Add fileinfo columns to dataframe.
            event_df = self._add_fileinfo(event_df, fileinfo)

            event_dfs.append(EventDataFrame(event_df))

        return event_dfs

    def pix2deg(self, verbose: bool = True) -> Dataset:
        """Compute gaze positions in degrees of visual angle from pixel coordinates.

        This method requires a properly initialized :py:attr:`~.Dataset.experiment` attribute.

        After success, the gaze dataframe is extended by the resulting dva columns.

        Parameters
        ----------
        verbose : bool
            If True, show progress of computation.

        Raises
        ------
        AttributeError
            If `gaze` is None or there are no gaze dataframes present in the `gaze` attribute, or
            if experiment is None.

        Returns
        -------
        Dataset
            Returns self, useful for method cascading.
        """
        self._check_gaze_dataframe()

        disable_progressbar = not verbose
        for gaze_df in tqdm(self.gaze, disable=disable_progressbar):
            gaze_df.pix2deg()

        return self

    def pos2vel(self, method: str = 'smooth', verbose: bool = True, **kwargs) -> Dataset:
        """Compute gaze velocites in dva/s from dva coordinates.

        This method requires a properly initialized :py:attr:`~.Dataset.experiment` attribute.

        After success, the gaze dataframe is extended by the resulting velocity columns.

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

        Returns
        -------
        Dataset
            Returns self, useful for method cascading.
        """
        self._check_gaze_dataframe()

        disable_progressbar = not verbose
        for gaze_df in tqdm(self.gaze, disable=disable_progressbar):
            gaze_df.pos2vel(method=method, **kwargs)

        return self

    def detect_events(
            self,
            method: EventDetectionCallable,
            eye: str | None = 'auto',
            clear: bool = False,
            verbose: bool = True,
            **kwargs,
    ) -> Dataset:
        """Detect events by applying a specific event detection method.

        Parameters
        ----------
        method : EventDetectionCallable
            The event detection method to be applied.
        eye : str
            Select which eye to choose. Valid options are ``auto``, ``left``, ``right`` or ``None``.
            If ``auto`` is passed, eye is inferred in the order ``['right', 'left', 'eye']`` from
            the available :py:attr:`~.Dataset.gaze` dataframe columns.
        clear : bool
            If ``True``, event DataFrame will be overwritten with new DataFrame instead of being
             merged into the existing one.
        verbose : bool
            If ``True``, show progress bar.
        **kwargs :
            Additional keyword arguments to be passed to the event detection method.

        Raises
        ------
        AttributeError
            If gaze files have not been loaded yet or gaze files do not contain the right columns.

        Returns
        -------
        Dataset
            Returns self, useful for method cascading.
        """
        self._check_gaze_dataframe()

        # Automatically infer eye to use for event detection.
        if eye == 'auto':
            if 'x_right_pos' in self.gaze[0].columns:
                eye = 'right'
            elif 'x_left_pos' in self.gaze[0].columns:
                eye = 'left'
            elif 'x_pos' in self.gaze[0].columns:
                eye = None
            else:
                raise AttributeError(
                    'Either right or left eye columns must be present in gaze data frame.'
                    f' Available columns are: {self.gaze[0].columns}',
                )

        if eye is None:
            position_columns = ['x_pos', 'y_pos']
            velocity_columns = ['x_vel', 'y_vel']
        else:
            position_columns = [f'x_{eye}_pos', f'y_{eye}_pos']
            velocity_columns = [f'x_{eye}_vel', f'y_{eye}_vel']

        if not set(position_columns).issubset(set(self.gaze[0].columns)):
            raise AttributeError(
                f'{eye} eye specified but required columns are not available in gaze dataframe.'
                f' required columns: {position_columns}'
                f', available columns: {self.gaze[0].columns}',
            )

        disable_progressbar = not verbose

        event_dfs: list[EventDataFrame] = []

        for gaze_df, fileinfo in tqdm(
                zip(self.gaze, self.fileinfo.to_dicts()), disable=disable_progressbar,
        ):

            positions = gaze_df.frame.select(position_columns).to_numpy()
            velocities = gaze_df.frame.select(velocity_columns).to_numpy()
            timesteps = gaze_df.frame.select('time').to_numpy()

            event_df = method(
                positions=positions, velocities=velocities, timesteps=timesteps, **kwargs,
            )

            event_df.frame = self._add_fileinfo(event_df.frame, fileinfo)
            event_dfs.append(event_df)

        if not self.events or clear:
            self.events = event_dfs
            return self

        for file_id, event_df in enumerate(event_dfs):
            self.events[file_id].frame = pl.concat(
                [self.events[file_id].frame, event_df.frame],
                how='diagonal',
            )
        return self

    def compute_event_properties(
            self,
            event_properties: str | list[str],
            verbose: bool = True,
    ) -> Dataset:
        """Calculate an event property for and add it as a column to the event dataframe.

        Parameters
        ----------
        event_properties:
            The event properties to compute.
        verbose : bool
            If ``True``, show progress bar.

        Raises
        ------
        InvalidProperty
            If ``property_name`` is not a valid property. See
            :py:mod:`pymovements.events.event_properties` for an overview of supported properties.

        Returns
        -------
        Dataset
            Returns self, useful for method cascading.
        """
        processor = EventGazeProcessor(event_properties)

        identifier_columns = [column for column in self.fileinfo.columns if column != 'filepath']

        disable_progressbar = not verbose
        for events, gaze in tqdm(zip(self.events, self.gaze), disable=disable_progressbar):
            new_properties = processor.process(events, gaze, identifiers=identifier_columns)

            new_properties = new_properties.drop(identifier_columns)
            new_properties = new_properties.drop(['name', 'onset', 'offset'])

            events.add_event_properties(new_properties)

        return self

    def clear_events(self) -> Dataset:
        """Clear event DataFrame.

        Returns
        -------
        Dataset
            Returns self, useful for method cascading.
        """
        if len(self.events) == 0:
            return self

        for file_id, _ in enumerate(self.events):
            self.events[file_id] = EventDataFrame()

        return self

    def save(
            self,
            events_dirname: str | None = None,
            preprocessed_dirname: str | None = None,
            verbose: int = 1,
            extension: str = 'feather',
    ) -> Dataset:
        """Save preprocessed gaze and event files.

        Data will be saved as feather/csv files to ``Dataset.preprocessed_roothpath`` or
        ``Dataset.events_roothpath`` with the same directory structure as the raw data.

        Parameters
        ----------
        events_dirname : str
            One-time usage of an alternative directory name to save data relative to dataset path.
            This argument is used only for this single call and does not alter
            :py:meth:`pymovements.Dataset.events_rootpath`.
        preprocessed_dirname : str
            One-time usage of an alternative directory name to save data relative to dataset path.
            This argument is used only for this single call and does not alter
            :py:meth:`pymovements.Dataset.preprocessed_rootpath`.
        verbose : int
            Verbosity level (0: no print output, 1: show progress bar, 2: print saved filepaths)
        extension:
            extension specifies the fileformat to store the data

        Returns
        -------
        Dataset
            Returns self, useful for method cascading.
        """
        self.save_events(events_dirname, verbose=verbose, extension=extension)
        self.save_preprocessed(preprocessed_dirname, verbose=verbose, extension=extension)
        return self

    def save_events(
        self, events_dirname: str | None = None,
        verbose: int = 1,
        extension: str = 'feather',
    ) -> Dataset:
        """Save events to files.

        Data will be saved as feather files to ``Dataset.events_roothpath`` with the same directory
        structure as the raw data.

        Parameters
        ----------
        events_dirname : str
            One-time usage of an alternative directory name to save data relative to dataset path.
            This argument is used only for this single call and does not alter
            :py:meth:`pymovements.Dataset.events_rootpath`.
        verbose : int
            Verbosity level (0: no print output, 1: show progress bar, 2: print saved filepaths)
        extension:
            Specifies the file format for loading data. Valid options are: `csv`, `feather`.
            :Default: `feather`.

        Raises
        ------
        ValueError
            If extension is not in list of valid extensions.

        Returns
        -------
        Dataset
            Returns self, useful for method cascading.
        """
        disable_progressbar = not verbose

        for file_id, event_df in enumerate(tqdm(self.events, disable=disable_progressbar)):
            raw_filepath = self.raw_rootpath / Path(self.fileinfo[file_id, 'filepath'])
            events_filepath = self._raw_to_event_filepath(
                raw_filepath, events_dirname=events_dirname,
                extension=extension,
            )

            event_df_out = event_df.frame.clone()
            for column in event_df_out.columns:
                if column in self.fileinfo.columns:
                    event_df_out = event_df_out.drop(column)

            if verbose >= 2:
                print('Save file to', events_filepath)

            events_filepath.parent.mkdir(parents=True, exist_ok=True)
            if extension == 'feather':
                event_df_out.write_ipc(events_filepath)
            elif extension == 'csv':
                event_df_out.write_csv(events_filepath)
            else:
                valid_extensions = ['csv', 'feather']
                raise ValueError(
                    f'unsupported file format "{extension}".'
                    f'Supported formats are: {valid_extensions}',
                )
        return self

    def save_preprocessed(
        self, preprocessed_dirname: str | None = None,
        verbose: int = 1,
        extension: str = 'feather',
    ) -> Dataset:
        """Save preprocessed gaze files.

        Data will be saved as feather files to ``Dataset.preprocessed_roothpath`` with the same
        directory structure as the raw data.

        Parameters
        ----------
        preprocessed_dirname : str
            One-time usage of an alternative directory name to save data relative to dataset path.
            This argument is used only for this single call and does not alter
            :py:meth:`pymovements.Dataset.preprocessed_rootpath`.
        verbose : int
            Verbosity level (0: no print output, 1: show progress bar, 2: print saved filepaths)
        extension:
            Specifies the file format for loading data. Valid options are: `csv`, `feather`.
            :Default: `feather`.

        Raises
        ------
        ValueError
            If extension is not in list of valid extensions.

        Returns
        -------
        Dataset
            Returns self, useful for method cascading.
        """
        disable_progressbar = not verbose

        for file_id, gaze_df in enumerate(tqdm(self.gaze, disable=disable_progressbar)):
            raw_filepath = self.raw_rootpath / Path(self.fileinfo[file_id, 'filepath'])
            preprocessed_filepath = self._raw_to_preprocessed_filepath(
                raw_filepath, preprocessed_dirname=preprocessed_dirname,
                extension=extension,
            )

            gaze_df_out = gaze_df.frame.clone()
            for column in gaze_df.columns:
                if column in self.fileinfo.columns:
                    gaze_df_out = gaze_df_out.drop(column)

            if verbose >= 2:
                print('Save file to', preprocessed_filepath)

            preprocessed_filepath.parent.mkdir(parents=True, exist_ok=True)
            if extension == 'feather':
                gaze_df_out.write_ipc(preprocessed_filepath)
            elif extension == 'csv':
                gaze_df_out.write_csv(preprocessed_filepath)
            else:
                valid_extensions = ['csv', 'feather']
                raise ValueError(
                    f'unsupported file format "{extension}".'
                    f'Supported formats are: {valid_extensions}',
                )
        return self

    @property
    def path(self) -> Path:
        """The path to the dataset directory.

        The dataset path points to the dataset directory under the root path. Per default the
        dataset path points to the exact same directory as the root path. Add ``dataset_dirname``
        to your initialization call to specify an explicit dataset directory in your root path.

        Example
        -------

        The default behavior is to locate the dataset in the root path without assuming a further
        dataset directory:
        >>> dataset = Dataset(root='/path/to/your/dataset', dataset_dirname='.')
        >>> dataset.path  # doctest: +SKIP
        Path('/path/to/your/dataset')

        To locate the dataset you can also specify an explicit dataset directory name in the root
        path:
        >>> dataset = Dataset(root='path/to/all/your/datasets/', dataset_dirname='your_dataset')
        >>> dataset.path  # doctest: +SKIP
        Path('path/to/all/your/datasets/your_dataset')
        """
        return self.root / self.dataset_dirname

    @property
    def root(self) -> Path:
        """The root path to your dataset.

        Example
        -------
        >>> dataset = Dataset(root='/path/to/your/dataset')
        >>> dataset.root  # doctest: +SKIP
        Path('/path/to/your/dataset')

        The root stays unaffected by the dataset directory name:
        >>> dataset = Dataset(root='/path/to/', dataset_dirname='your_dataset')
        >>> dataset.root  # doctest: +SKIP
        Path('/path/to')
        """
        return self._root

    @property
    def events_rootpath(self) -> Path:
        """The path to the directory of the event data.

        The path points to the events directory under the root path.

        Example
        -------

        >>> dataset = Dataset(root='/path/to/your/datasets/', dataset_dirname='your_dataset')
        >>> dataset.events_rootpath  # doctest: +SKIP
        Path('/path/to/your/datasets/your_dataset/events')

        You can also explicitely specify the event directory name. The default is `events`.
        >>> dataset = Dataset(
        ...     root='/path/to/your/datasets/',
        ...     dataset_dirname='your_dataset',
        ...     events_dirname='your_events',
        ... )
        >>> dataset.events_rootpath  # doctest: +SKIP
        Path('/path/to/your/datasets/your_dataset/your_events')
        """
        return self.path / self.events_dirname

    @property
    def preprocessed_rootpath(self) -> Path:
        """The path to the directory of the preprocessed gaze data.

        The path points to the preprocessed data directory under the root path.

        Example
        -------

        >>> dataset = Dataset(root='/path/to/your/datasets/', dataset_dirname='your_dataset')
        >>> dataset.preprocessed_rootpath  # doctest: +SKIP
        Path('/path/to/your/datasets/your_dataset/preprocessed')

        You can also explicitely specify the preprocessed directory name. The default is
        `preprocessed`.
        >>> dataset = Dataset(
        ...     root='/path/to/your/datasets/',
        ...     dataset_dirname='your_dataset',
        ...     preprocessed_dirname='your_preprocessed_data',
        ... )
        >>> dataset.preprocessed_rootpath  # doctest: +SKIP
        Path('/path/to/your/datasets/your_dataset/your_preprocessed_data')
        """
        return self.path / self.preprocessed_dirname

    @property
    def raw_rootpath(self) -> Path:
        """The path to the directory of the raw data.

        The path points to the raw data directory under the root path.

        Example
        -------

        >>> dataset = Dataset(root='/path/to/your/datasets/', dataset_dirname='your_dataset')
        >>> dataset.preprocessed_rootpath  # doctest: +SKIP
        Path('/path/to/your/datasets/your_dataset/raw')

        You can also explicitely specify the preprocessed directory name. The default is `raw`.
        >>> dataset = Dataset(
        ...     root='/path/to/your/datasets/',
        ...     dataset_dirname='your_dataset',
        ...     raw_dirname='your_raw_data',
        ... )
        >>> dataset.raw_rootpath  # doctest: +SKIP
        Path('/path/to/your/datasets/your_dataset/your_raw_data')

        If your raw data is not in a separate directory under the root path then you can also
        specify `.` as the directory name. We discourage this and advise the user to keep raw data
        and preprocessed data separated.
        ... dataset = Dataset(
        >>> dataset = Dataset(
        ...     root='/path/to/your/datasets/',
        ...     dataset_dirname='your_dataset',
        ...     raw_dirname='.',
        ... )
        >>> dataset.raw_rootpath  # doctest: +SKIP
        Path('/path/to/your/datasets/your_dataset/your_raw_data')
        """
        return self.path / self.raw_dirname

    def _check_fileinfo(self) -> None:
        """Check if fileinfo attribute is set and there is at least one row present."""
        if self.fileinfo is None:
            raise AttributeError(
                'fileinfo was not loaded yet. please run load() or infer_fileinfo() beforehand',
            )
        if len(self.fileinfo) == 0:
            raise AttributeError('no files present in fileinfo attribute')

    def _check_gaze_dataframe(self) -> None:
        """Check if gaze attribute is set and there is at least one gaze dataframe available."""
        if self.gaze is None:
            raise AttributeError('gaze files were not loaded yet. please run load() beforehand')
        if len(self.gaze) == 0:
            raise AttributeError('no files present in gaze attribute')

    def _raw_to_preprocessed_filepath(
            self,
            raw_filepath: Path,
            preprocessed_dirname: str | None = None,
            extension: str = 'feather',
    ) -> Path:
        """Get preprocessed filepath in accordance to filepath of the raw file.

        The preprocessed filepath will point to a feather file.

        Parameters
        ----------
        raw_filepath : Path
            The Path to the raw file.
        preprocessed_dirname : str
            One-time usage of an alternative directory name to save data relative to
            :py:meth:`pymovements.Dataset.path`.
            This argument is used only for this single call and does not alter
            :py:meth:`pymovements.Dataset.preprocessed_rootpath`.
        extension:
            extension specifies the fileformat to store the data

        Returns
        -------
        Path
            The Path to the preprocessed feather file.
        """
        relative_raw_dirpath = raw_filepath.parent
        relative_raw_dirpath = relative_raw_dirpath.relative_to(self.raw_rootpath)

        if preprocessed_dirname is None:
            preprocessed_rootpath = self.preprocessed_rootpath
        else:
            preprocessed_rootpath = self.path / preprocessed_dirname

        preprocessed_file_dirpath = preprocessed_rootpath / relative_raw_dirpath

        # Get new filename for saved feather file.
        preprocessed_filename = raw_filepath.stem + '.' + extension

        return preprocessed_file_dirpath / preprocessed_filename

    def _raw_to_event_filepath(
        self,
        raw_filepath: Path,
        events_dirname: str | None = None,
        extension: str = 'feather',
    ) -> Path:
        """Get event filepath in accordance to filepath of the raw file.

        The event filepath will point to file with the specified extension.

        Parameters
        ----------
        raw_filepath : Path
            The Path to the raw file.
        events_dirname : str
            One-time usage of an alternative directory name to save data relative to
            :py:meth:`pymovements.Dataset.path`.
            This argument is used only for this single call and does not alter
            :py:meth:`pymovements.Dataset.events_rootpath`.
        extension:
            extension specifies the fileformat to store the data

        Returns
        -------
        Path
            The Path to the event feather file.
        """
        relative_raw_dirpath = raw_filepath.parent
        relative_raw_dirpath = relative_raw_dirpath.relative_to(self.raw_rootpath)

        if events_dirname is None:
            events_rootpath = self.events_rootpath
        else:
            events_rootpath = self.path / events_dirname

        events_file_dirpath = events_rootpath / relative_raw_dirpath

        # Get new filename for saved feather file.
        events_filename = raw_filepath.stem + '.' + extension

        return events_file_dirpath / events_filename

    def _add_fileinfo(self, df: pl.DataFrame, fileinfo: dict[str, Any]) -> pl.DataFrame:
        """Add columns from fileinfo to dataframe.

        Parameters
        ----------
        df : pl.DataFrame
            Base dataframe to add fileinfo to.
        fileinfo : dict[str, Any]
            Dictionary of fileinfo row.

        Returns
        -------
        pl.DataFrame:
            Dataframe with added columns from fileinfo dictionary keys.
        """

        df = df.select(
            [
                pl.lit(value).alias(column)
                for column, value in fileinfo.items()
                if column != 'filepath'
            ] + [pl.all()],
        )

        # Cast columns from fileinfo according to specification.
        df = df.with_columns([
            pl.col(fileinfo_key).cast(fileinfo_dtype)
            for fileinfo_key, fileinfo_dtype in self._filename_regex_dtypes.items()
        ])
        return df
