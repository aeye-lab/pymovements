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

from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from typing import Any

import polars as pl
from tqdm.auto import tqdm

from pymovements.dataset import dataset_download
from pymovements.dataset import dataset_files
from pymovements.dataset.dataset_definition import DatasetDefinition
from pymovements.dataset.dataset_library import DatasetLibrary
from pymovements.dataset.dataset_paths import DatasetPaths
from pymovements.events.frame import EventDataFrame
from pymovements.events.processing import EventGazeProcessor
from pymovements.gaze import GazeDataFrame


class Dataset:
    """Dataset base class."""

    def __init__(
            self,
            definition: str | DatasetDefinition | type[DatasetDefinition],
            path: str | Path | DatasetPaths,
    ):
        """Initialize the dataset object.

        Parameters
        ----------
        definition : str, DatasetDefinition
            Dataset definition to initialize dataset with.
        path : DatasetPaths, optional
            Path to the dataset directory. You can set up a custom directory structure by passing a
            :py:class:`~pymovements.DatasetPaths` instance.
        """
        self.fileinfo: pl.DataFrame = pl.DataFrame()
        self.gaze: list[GazeDataFrame] = []
        self.events: list[EventDataFrame] = []

        if isinstance(definition, str):
            definition = DatasetLibrary.get(definition)()
        if isinstance(definition, type):
            definition = definition()
        self.definition = deepcopy(definition)

        if isinstance(path, (str, Path)):
            self.paths = DatasetPaths(root=path, dataset='.')
        else:
            self.paths = deepcopy(path)
        # Fill dataset directory name with dataset definition name if specified.
        self.paths.fill_name(self.definition.name)

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
            present in the fileinfo dataframe inferred by `scan()`. Values can be either
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
        self.scan()
        self.fileinfo = dataset_files.take_subset(fileinfo=self.fileinfo, subset=subset)

        self.load_gaze_files(
            preprocessed=preprocessed, preprocessed_dirname=preprocessed_dirname,
            extension=extension,
        )

        if events:
            self.load_event_files(
                events_dirname=events_dirname,
                extension=extension,
            )

        return self

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

    def load_gaze_files(
            self,
            preprocessed: bool = False,
            preprocessed_dirname: str | None = None,
            extension: str = 'feather',
    ) -> Dataset:
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
        Dataset
            Returns self, useful for method cascading.

        Raises
        ------
        AttributeError
            If `fileinfo` is None or the `fileinfo` dataframe is empty.
        RuntimeError
            If file type of gaze file is not supported.
        """
        self._check_fileinfo()
        self.gaze = dataset_files.load_gaze_files(
            definition=self.definition,
            fileinfo=self.fileinfo,
            paths=self.paths,
            preprocessed=preprocessed,
            preprocessed_dirname=preprocessed_dirname,
            extension=extension,
        )
        return self

    def load_event_files(
            self,
            events_dirname: str | None = None,
            extension: str = 'feather',
    ) -> Dataset:
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
        self.events = dataset_files.load_event_files(
            definition=self.definition,
            fileinfo=self.fileinfo,
            paths=self.paths,
            events_dirname=events_dirname,
            extension=extension,
        )
        return self

    def apply(
            self,
            function: str,
            *,
            verbose: bool = True,
            **kwargs: Any,
    ) -> Dataset:
        """Apply preprocessing method to all GazeDataFrames in Dataset.

        Parameters
        ----------
        function: str
            Name of the preprocessing function to apply.
        verbose : bool
            If True, show progress bar of computation.
        kwargs:
            kwargs that will be forwarded when calling the preprocessing method.

        Examples
        --------
        Let's load in our dataset first,
        >>> import pymovements as pm
        >>>
        >>> dataset = pm.Dataset("ToyDataset", path='toy_dataset')
        >>> dataset.download()# doctest:+ELLIPSIS
        Downloading ... to toy_dataset...downloads...
        Checking integrity of ...
        Extracting ... to toy_dataset...raw
        <pymovements.dataset.dataset.Dataset object at ...>
        >>> dataset.load()# doctest:+ELLIPSIS
        <pymovements.dataset.dataset.Dataset object at ...>

        Use apply for your gaze transformations:
        >>> dataset.apply('pix2deg')# doctest:+ELLIPSIS
        <pymovements.dataset.dataset.Dataset object at ...>

        >>> dataset.apply('pos2vel', method='neighbors')# doctest:+ELLIPSIS
        <pymovements.dataset.dataset.Dataset object at ...>

        Use apply for your event detection:
        >>> dataset.apply('ivt')# doctest:+ELLIPSIS
        <pymovements.dataset.dataset.Dataset object at ...>

        >>> dataset.apply('microsaccades', minimum_duration=8)# doctest:+ELLIPSIS
        <pymovements.dataset.dataset.Dataset object at ...>
        """
        self._check_gaze_dataframe()

        disable_progressbar = not verbose
        for gaze in tqdm(self.gaze, disable=disable_progressbar):
            gaze.apply(function, **kwargs)

        return self

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
        return self.apply('pix2deg', verbose=verbose)

    def pos2acc(
            self,
            *,
            degree: int = 2,
            window_length: int = 7,
            padding: str | float | int | None = 'nearest',
            verbose: bool = True,
    ) -> Dataset:
        """Compute gaze accelerations in dva/s^2 from dva coordinates.

        This method requires a properly initialized :py:attr:`~.Dataset.experiment` attribute.

        After success, the gaze dataframe is extended by the resulting acceleration columns.

        Parameters
        ----------
        window_length:
            The window size to use.
        degree:
            The degree of the polynomial to use.
        padding:
            The padding method to use. See ``savitzky_golay`` for details.
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
        return self.apply(
            'pos2acc',
            window_length=window_length,
            degree=degree,
            padding=padding,
            verbose=verbose,
        )

    def pos2vel(
            self,
            method: str = 'fivepoint',
            *,
            verbose: bool = True,
            **kwargs: Any,
    ) -> Dataset:
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
        return self.apply('pos2vel', method=method, verbose=verbose, **kwargs)

    def detect_events(
            self,
            method: Callable[..., EventDataFrame] | str,
            *,
            eye: str = 'auto',
            clear: bool = False,
            verbose: bool = True,
            **kwargs: Any,
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
        return self.detect(
            method=method,
            eye=eye,
            clear=clear,
            verbose=verbose,
            **kwargs,
        )

    def detect(
            self,
            method: Callable[..., EventDataFrame] | str,
            *,
            eye: str = 'auto',
            clear: bool = False,
            verbose: bool = True,
            **kwargs: Any,
    ) -> Dataset:
        """Detect events by applying a specific event detection method.

        Alias for :py:meth:`pymovements.Dataset.detect_events`

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

        if not self.events:
            self.events = [gaze.events for gaze in self.gaze]

        disable_progressbar = not verbose
        for file_id, (gaze, fileinfo_row) in tqdm(
                enumerate(zip(self.gaze, self.fileinfo.to_dicts())), disable=disable_progressbar,
        ):
            gaze.detect(method, eye=eye, clear=clear, **kwargs)
            # workaround until events are fully part of the GazeDataFrame
            gaze.events.frame = dataset_files.add_fileinfo(
                definition=self.definition,
                df=gaze.events.frame,
                fileinfo=fileinfo_row,
            )
            self.events[file_id] = gaze.events
        return self

    def compute_event_properties(
            self,
            event_properties: str | tuple[str, dict[str, Any]]
            | list[str | tuple[str, dict[str, Any]]],
            name: str | None = None,
            verbose: bool = True,
    ) -> Dataset:
        """Calculate an event property for and add it as a column to the event dataframe.

        Parameters
        ----------
        event_properties:
            The event properties to compute.
        name:
            Process only events that match the name.
        verbose : bool
            If ``True``, show progress bar.

        Raises
        ------
        InvalidProperty
            If ``property_name`` is not a valid property. See
            :py:mod:`pymovements.events.event_properties` for an overview of supported properties.
        RuntimeError
            If specified event name ``name`` is missing from ``events``.

        Returns
        -------
        Dataset
            Returns self, useful for method cascading.
        """
        processor = EventGazeProcessor(event_properties)

        identifier_columns = [column for column in self.fileinfo.columns if column != 'filepath']

        disable_progressbar = not verbose
        for events, gaze in tqdm(zip(self.events, self.gaze), disable=disable_progressbar):
            new_properties = processor.process(
                events, gaze, identifiers=identifier_columns, name=name,
            )
            join_on = identifier_columns + ['name', 'onset', 'offset']
            events.add_event_properties(new_properties, join_on=join_on)

        return self

    def compute_properties(
            self,
            event_properties: str | tuple[str, dict[str, Any]]
            | list[str | tuple[str, dict[str, Any]]],
            name: str | None = None,
            verbose: bool = True,
    ) -> Dataset:
        """Calculate an event property for and add it as a column to the event dataframe.

        Alias for :py:meth:`pymovements.Dataset.compute_event_properties`

        Parameters
        ----------
        event_properties:
            The event properties to compute.
        name:
            Process only events that match the name.
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
        return self.compute_event_properties(
            event_properties=event_properties,
            name=name,
            verbose=verbose,
        )

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
            self,
            events_dirname: str | None = None,
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
        dataset_files.save_events(
            events=self.events,
            fileinfo=self.fileinfo,
            paths=self.paths,
            events_dirname=events_dirname,
            verbose=verbose,
            extension=extension,
        )
        return self

    def save_preprocessed(
            self,
            preprocessed_dirname: str | None = None,
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
        dataset_files.save_preprocessed(
            gaze=self.gaze,
            fileinfo=self.fileinfo,
            paths=self.paths,
            preprocessed_dirname=preprocessed_dirname,
            verbose=verbose,
            extension=extension,
        )
        return self

    def download(
            self,
            *,
            extract: bool = True,
            remove_finished: bool = False,
            verbose: int = 1,
    ) -> Dataset:
        """Download dataset resources.

        This downloads all resources of the dataset. Per default this also extracts all archives
        into :py:meth:`Dataset.paths.raw`,
        To save space on your device you can remove the archive files after
        successful extraction with ``remove_finished=True``.

        If a corresponding file already exists in the local system, its checksum is calculated and
        checked against the expected checksum.
        Downloading will be evaded if the integrity of the existing file can be verified.
        If the existing file does not match the expected checksum it is overwritten with the
        downloaded new file.

        Parameters
        ----------
        extract : bool
            Extract dataset archive files.
        remove_finished : bool
            Remove archive files after extraction.
        verbose : int
            Verbosity levels: (1) Show download progress bar and print info messages on downloading
            and extracting archive files without printing messages for recursive archive extraction.
            (2) Print additional messages for each recursive archive extract.

        Raises
        ------
        AttributeError
            If number of mirrors or number of resources specified for dataset is zero.
        RuntimeError
            If downloading a resource failed for all given mirrors.

        Returns
        -------
        PublicDataset
            Returns self, useful for method cascading.
        """
        dataset_download.download_dataset(
            definition=self.definition,
            paths=self.paths,
            extract=extract,
            remove_finished=remove_finished,
            verbose=bool(verbose),
        )
        return self

    def extract(self, remove_finished: bool = False, verbose: int = 1) -> Dataset:
        """Extract downloaded dataset archive files.

        Parameters
        ----------
        remove_finished : bool
            Remove archive files after extraction.
        verbose : int
            Verbosity levels: (1) Print messages for extracting each dataset resource without
            printing messages for recursive archives. (2) Print additional messages for each
            recursive archive extract.

        Returns
        -------
        PublicDataset
            Returns self, useful for method cascading.
        """
        dataset_download.extract_dataset(
            definition=self.definition,
            paths=self.paths,
            remove_finished=remove_finished,
            verbose=verbose,
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
        By passing a `str` or a `Path` as `path` during initialization you can explicitly set the
        directory path of the dataset:
        >>> import pymovements as pm
        >>>
        >>> dataset = pm.Dataset("ToyDataset", path='/path/to/your/dataset')
        >>> dataset.path# doctest: +SKIP
        Path('/path/to/your/dataset')

        If you just want to specify the root directory path which holds all your local datasets, you
        can create pass a :py:class:`~pymovements.DatasetPaths` object and set the `root`:
        >>> paths = pm.DatasetPaths(root='/path/to/your/common/root/')
        >>> dataset = pm.Dataset("ToyDataset", path=paths)
        >>> dataset.path# doctest: +SKIP
        Path('/path/to/your/common/root/ToyDataset')

        You can also specify an alternative dataset directory name:
        >>> paths = pm.DatasetPaths(root='/path/to/your/common/root/', dataset='my_dataset')
        >>> dataset = pm.Dataset("ToyDataset", path=paths)
        >>> dataset.path# doctest: +SKIP
        Path('/path/to/your/common/root/my_dataset')
        """
        return self.paths.dataset

    def _check_fileinfo(self) -> None:
        """Check if fileinfo attribute is set and there is at least one row present."""
        if self.fileinfo is None:
            raise AttributeError(
                'fileinfo was not loaded yet. please run load() or scan() beforehand',
            )
        if len(self.fileinfo) == 0:
            raise AttributeError('no files present in fileinfo attribute')

    def _check_gaze_dataframe(self) -> None:
        """Check if gaze attribute is set and there is at least one gaze dataframe available."""
        if self.gaze is None:
            raise AttributeError('gaze files were not loaded yet. please run load() beforehand')
        if len(self.gaze) == 0:
            raise AttributeError('no files present in gaze attribute')
