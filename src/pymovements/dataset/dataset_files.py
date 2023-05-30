# Copyright (c) 2023 The pymovements Project Authors
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
"""Functionality to scan, load and save dataset files."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl
from tqdm.auto import tqdm

from pymovements.dataset.dataset_definition import DatasetDefinition
from pymovements.dataset.dataset_paths import DatasetPaths
from pymovements.events.events import EventDataFrame
from pymovements.gaze.gaze_dataframe import GazeDataFrame
from pymovements.utils.parsing import parse_eyelink
from pymovements.utils.paths import match_filepaths
from pymovements.utils.strings import curly_to_regex


def scan_dataset(definition: DatasetDefinition, paths: DatasetPaths) -> pl.DataFrame:
    """Infer information from filepaths and filenames.

    Parameters
    ----------
    definition
        The dataset definition.
    paths
        The dataset paths.

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
        path=paths.raw,
        regex=curly_to_regex(definition.filename_format),
        relative=True,
    )

    if len(fileinfo_dicts) == 0:
        print(definition.filename_format)
        print(curly_to_regex(definition.filename_format))
        raise RuntimeError(f'no matching files found in {paths.raw}')

    # Create dataframe from all fileinfo records.
    fileinfo_df = pl.from_dicts(data=fileinfo_dicts, infer_schema_length=1)
    fileinfo_df = fileinfo_df.sort(by='filepath')

    fileinfo_df = fileinfo_df.with_columns([
        pl.col(fileinfo_key).cast(fileinfo_dtype)
        for fileinfo_key, fileinfo_dtype in definition.filename_format_dtypes.items()
    ])

    return fileinfo_df


def load_event_files(
        definition: DatasetDefinition,
        fileinfo: pl.DataFrame,
        paths: DatasetPaths,
        events_dirname: str | None = None,
        extension: str = 'feather',
) -> list[EventDataFrame]:
    """Load all event files according to fileinfo dataframe.

    Parameters
    ----------
    definition:
        The dataset definition.
    fileinfo:
        A dataframe holding file information.
    paths:
        Path of directory containing event files.
    events_dirname : str
        One-time usage of an alternative directory name to save data relative to dataset path.
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
    event_dfs: list[EventDataFrame] = []

    # read and preprocess input files
    for fileinfo_row in tqdm(fileinfo.to_dicts()):
        filepath = Path(fileinfo_row['filepath'])
        filepath = paths.raw / filepath

        filepath = paths.raw_to_event_filepath(
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
        event_df = add_fileinfo(
            definition=definition,
            df=event_df,
            fileinfo=fileinfo_row,
        )

        event_dfs.append(EventDataFrame(event_df))

    return event_dfs


def load_gaze_files(
        definition: DatasetDefinition,
        fileinfo: pl.DataFrame,
        paths: DatasetPaths,
        preprocessed: bool = False,
        preprocessed_dirname: str | None = None,
        extension: str = 'feather',
) -> list[GazeDataFrame]:
    """Load all available gaze data files.

    Parameters
    ----------
    definition:
        The dataset definition.
    fileinfo:
        A dataframe holding file information.
    paths:
        Path of directory containing event files.
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
    gaze_dfs: list[GazeDataFrame] = []

    # Read gaze files from fileinfo attribute.
    for fileinfo_row in tqdm(fileinfo.to_dicts()):
        filepath = Path(fileinfo_row['filepath'])
        filepath = paths.raw / filepath

        if preprocessed:
            filepath = paths.get_preprocessed_filepath(
                filepath, preprocessed_dirname=preprocessed_dirname,
                extension=extension,
            )

        gaze_df = load_gaze_file(
            filepath=filepath,
            preprocessed=preprocessed,
            custom_read_kwargs=definition.custom_read_kwargs,
        )

        # Add fileinfo columns to dataframe.
        gaze_df = add_fileinfo(
            definition=definition,
            df=gaze_df,
            fileinfo=fileinfo_row,
        )

        gaze_dfs.append(GazeDataFrame(gaze_df, experiment=definition.experiment))

    return gaze_dfs


def load_gaze_file(
        filepath: Path,
        preprocessed: bool = False,
        custom_read_kwargs: dict[str, Any] | None = None,
) -> pl.DataFrame:
    """Load a gaze data file as a polars DataFrame.

    Parameters
    ----------
    filepath:
        Path of gaze file.
    preprocessed:
        If ``True``, saved preprocessed data will be loaded, otherwise raw data will be loaded.
    custom_read_kwargs:
        Dictionary of keyword arguments for reading gaze file.

    Returns
    -------
    pl.DataFrame
        The resulting polars.DataFrame

    Raises
    ------
    RuntimeError
        If file type of gaze file is not supported.
    ValueError
        If extension is not in list of valid extensions.
    """
    if custom_read_kwargs is None:
        custom_read_kwargs = {}

    if filepath.suffix == '.csv':
        if preprocessed:
            gaze_df = pl.read_csv(filepath)
        else:
            gaze_df = pl.read_csv(filepath, **custom_read_kwargs)
    elif filepath.suffix == '.feather':
        gaze_df = pl.read_ipc(filepath)
    elif filepath.suffix == '.asc':
        gaze_df = parse_eyelink(filepath, **custom_read_kwargs)
    else:
        valid_extensions = ['csv', 'feather', 'asc']
        raise ValueError(
            f'unsupported file format "{filepath.suffix}".'
            f'Supported formats are: {valid_extensions}',
        )

    return gaze_df


def add_fileinfo(
        definition: DatasetDefinition,
        df: pl.DataFrame,
        fileinfo: dict[str, Any],
) -> pl.DataFrame:
    """Add columns from fileinfo to dataframe.

    Parameters
    ----------
    definition
        The dataset definition.
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
        for fileinfo_key, fileinfo_dtype in definition.filename_format_dtypes.items()
    ])
    return df


def save_events(
        events: list[EventDataFrame],
        fileinfo: pl.DataFrame,
        paths: DatasetPaths,
        events_dirname: str | None = None,
        verbose: int = 1,
        extension: str = 'feather',
) -> None:
    """Save events to files.

    Data will be saved as feather files to ``Dataset.events_roothpath`` with the same directory
    structure as the raw data.

    Parameters
    ----------
    events:
        The event dataframes to save.
    fileinfo:
        A dataframe holding file information.
    paths:
        Path of directory containing event files.
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
    """
    disable_progressbar = not verbose

    for file_id, event_df in enumerate(tqdm(events, disable=disable_progressbar)):
        raw_filepath = paths.raw / Path(fileinfo[file_id, 'filepath'])
        events_filepath = paths.raw_to_event_filepath(
            raw_filepath, events_dirname=events_dirname,
            extension=extension,
        )

        event_df_out = event_df.frame.clone()
        for column in event_df_out.columns:
            if column in fileinfo.columns:
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


def save_preprocessed(
        gaze: list[GazeDataFrame],
        fileinfo: pl.DataFrame,
        paths: DatasetPaths,
        preprocessed_dirname: str | None = None,
        verbose: int = 1,
        extension: str = 'feather',
) -> None:
    """Save preprocessed gaze files.

    Data will be saved as feather files to ``Dataset.preprocessed_roothpath`` with the same
    directory structure as the raw data.

    Parameters
    ----------
    gaze:
        The gaze dataframes to save.
    fileinfo:
        A dataframe holding file information.
    paths:
        Path of directory containing event files.
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
    """
    disable_progressbar = not verbose

    for file_id, gaze_df in enumerate(tqdm(gaze, disable=disable_progressbar)):
        raw_filepath = paths.raw / Path(fileinfo[file_id, 'filepath'])
        preprocessed_filepath = paths.get_preprocessed_filepath(
            raw_filepath, preprocessed_dirname=preprocessed_dirname,
            extension=extension,
        )

        gaze_df_out = gaze_df.frame.clone()
        for column in gaze_df.columns:
            if column in fileinfo.columns:
                gaze_df_out = gaze_df_out.drop(column)

        if verbose >= 2:
            print('Save file to', preprocessed_filepath)

        preprocessed_filepath.parent.mkdir(parents=True, exist_ok=True)
        if extension == 'feather':
            gaze_df_out.write_ipc(preprocessed_filepath)
        elif extension == 'csv':
            gaze_df_out.write_csv(preprocessed_filepath)
        else:
            valid_extensions = ['csv', 'feather', 'asc']
            raise ValueError(
                f'unsupported file format "{extension}".'
                f'Supported formats are: {valid_extensions}',
            )


def take_subset(
        fileinfo: pl.DataFrame,
        subset: None | dict[
            str, bool | float | int | str | list[bool | float | int | str],
        ] = None,
) -> pl.DataFrame:
    """Take a subset of the fileinfo dataframe.

    Parameters
    ----------
    fileinfo : pl.DataFrame
        File information dataframe.
    subset : dict, optional
        If specified, take a subset of the dataset. All keys in the dictionary must be
        present in the fileinfo dataframe inferred by `scan_dataset()`. Values can be either
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
        elif isinstance(subset_value, (list, tuple, range)):
            column_values = subset_value
        else:
            raise TypeError(
                f'subset values must be of type bool, float, int, str, range, or list, '
                f'but value of pair {subset_key}: {subset_value} is of type {type(subset_value)}',
            )

        fileinfo = fileinfo.filter(pl.col(subset_key).is_in(column_values))
    return fileinfo
