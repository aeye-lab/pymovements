# Copyright (c) 2023-2025 The pymovements Project Authors
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
"""DatasetDefinition module."""
from __future__ import annotations

import builtins
import importlib
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any

import yaml

from pymovements.dataset._utils._yaml import type_constructor
from pymovements.gaze.experiment import Experiment
from pymovements.gaze.eyetracker import EyeTracker
from pymovements.gaze.screen import Screen


yaml.add_multi_constructor('!', type_constructor, Loader=yaml.SafeLoader)


@dataclass
class DatasetDefinition:
    """Definition to initialize a :py:class:`~pymovements.dataset.Dataset`.

    Attributes
    ----------
    name: str
        The name of the dataset. (default: '.')
    has_files: dict[str, bool]
        Indicate whether the dataset contains 'gaze', 'precomputed_events', and
        'precomputed_reading_measures'.
    mirrors: dict[str, list[str]] | dict[str, tuple[str, ...]]
        A list of mirrors of the dataset. Each entry must be of type `str` and end with a '/'.
        (default: field(default_factory=dict))
    resources: dict[str, list[dict[str, str]]] | dict[str, tuple[dict[str, str], ...]]
        A list of dataset resources. Each list entry must be a dictionary with the following keys:
        - `resource`: The url suffix of the resource. This will be concatenated with the mirror.
        - `filename`: The filename under which the file is saved as.
        - `md5`: The MD5 checksum of the respective file.
        (default: field(default_factory=dict))
    experiment: Experiment | None
        The experiment definition. (default: None)
    extract: dict[str, bool]
        Decide whether to extract the data.
    filename_format: dict[str, str]
        Regular expression which will be matched before trying to load the file. Namedgroups will
        appear in the `fileinfo` dataframe. (default: field(default_factory=dict))
    filename_format_schema_overrides: dict[str, dict[str, type]]
        If named groups are present in the `filename_format`, this makes it possible to cast
        specific named groups to a particular datatype. (default: field(default_factory=dict))
    custom_read_kwargs: dict[str, dict[str, Any]]
        If specified, these keyword arguments will be passed to the file reading function. The
        behavior of this argument depends on the file extension of the dataset files.
        If the file extension is `.csv` the keyword arguments will be passed
        to :py:func:`polars.read_csv`. If the file extension is`.asc` the keyword arguments
        will be passed to :py:func:`pymovements.utils.parsing.parse_eyelink`.
        See Notes for more details on how to use this argument.
        (default: field(default_factory=dict))
    column_map : dict[str, str]
        The keys are the columns to read, the values are the names to which they should be renamed.
        (default: field(default_factory=dict))
    trial_columns: list[str] | None
            The name of the trial columns in the input data frame. If the list is empty or None,
            the input data frame is assumed to contain only one trial. If the list is not empty,
            the input data frame is assumed to contain multiple trials and the transformation
            methods will be applied to each trial separately. (default: None)
    time_column: str | None
        The name of the timestamp column in the input data frame. This column will be renamed to
        ``time``. (default: None)

    time_unit: str | None
        The unit of the timestamps in the timestamp column in the input data frame. Supported
        units are 's' for seconds, 'ms' for milliseconds and 'step' for steps. If the unit is
        'step' the experiment definition must be specified. All timestamps will be converted to
        milliseconds. (default: 'ms')

    pixel_columns: list[str] | None
        The name of the pixel position columns in the input data frame. These columns will be
        nested into the column ``pixel``. If the list is empty or None, the nested ``pixel``
        column will not be created. (default: None)
    position_columns: list[str] | None
        The name of the dva position columns in the input data frame. These columns will be
        nested into the column ``position``. If the list is empty or None, the nested
        ``position`` column will not be created. (default: None)
    velocity_columns: list[str] | None
        The name of the velocity columns in the input data frame. These columns will be nested
        into the column ``velocity``. If the list is empty or None, the nested ``velocity``
        column will not be created. (default: None)
    acceleration_columns: list[str] | None
        The name of the acceleration columns in the input data frame. These columns will be
        nested into the column ``acceleration``. If the list is empty or None, the nested
        ``acceleration`` column will not be created. (default: None)
    distance_column : str | None
        The name of the column containing eye-to-screen distance in millimeters for each sample
        in the input data frame. If specified, the column will be used for pixel to dva
        transformations. If not specified, the constant eye-to-screen distance will be taken from
        the experiment definition. This column will be renamed to ``distance``. (default: None)

    Notes
    -----
    When working with the ``gaze_custom_read_kwargs`` attribute there are specific use cases and
    considerations to keep in mind, especially for reading csv files:

    1. Custom separator
    To read a csv file with a custom separator, you can pass the `separator` keyword argument to
    ``gaze_custom_read_kwargs``. For example pass ``gaze_custom_read_kwargs={'separator': ';'}`` to
    read a semicolon-separated csv file.

    2. Reading subset of columns
    To read only specific columns, specify them in ``gaze_custom_read_kwargs``. For example:
    ``gaze_custom_read_kwargs={'columns': ['col1', 'col2']}``

    3. Specifying column datatypes
    ``polars.read_csv`` infers data types from a fixed number of rows, which might not be accurate
    for the entire dataset. To ensure correct data types, you can pass a dictionary to the
    ``schema_overrides`` keyword argument in ``gaze_custom_read_kwargs``.
    Use data types from the `polars` library.
    For instance:
    ``gaze_custom_read_kwargs={'schema_overrides': {'col1': polars.Int64, 'col2': polars.Float64}}``
    """

    # pylint: disable=too-many-instance-attributes
    name: str = '.'
    has_files: dict[str, bool] = field(default_factory=dict)

    mirrors: dict[str, list[str]] | dict[str, tuple[str, ...]] = field(default_factory=dict)

    resources: dict[str, list[dict[str, str]]] | dict[str, tuple[dict[str, str], ...]] = field(
        default_factory=dict,
    )

    experiment: Experiment | None = field(default_factory=Experiment)

    extract: dict[str, bool] = field(default_factory=dict)

    filename_format: dict[str, str] = field(default_factory=dict)

    filename_format_schema_overrides: dict[str, dict[str, type]] = field(default_factory=dict)

    custom_read_kwargs: dict[str, dict[str, Any]] = field(default_factory=dict)

    column_map: dict[str, str] = field(default_factory=dict)

    trial_columns: list[str] | None = None
    time_column: str | None = None
    time_unit: str | None = 'ms'
    pixel_columns: list[str] | None = None
    position_columns: list[str] | None = None
    velocity_columns: list[str] | None = None
    acceleration_columns: list[str] | None = None
    distance_column: str | None = None

    @staticmethod
    def from_yaml(path: str | Path) -> DatasetDefinition:
        """Load a dataset definition from a YAML file.

        Parameters
        ----------
        path: str | Path
            Path to the YAML definition file

        Returns
        -------
        DatasetDefinition
            Initialized dataset definition
        """
        with open(path, encoding='utf-8') as f:
            data = yaml.safe_load(f)

        # Convert experiment dict to Experiment object if present
        if 'experiment' in data:
            if 'eyetracker' in data['experiment']:
                eyetracker = EyeTracker(**data['experiment'].pop('eyetracker'))
            else:
                eyetracker = None
            if 'screen' in data['experiment']:
                screen = Screen(**data['experiment'].pop('screen'))
            else:
                screen = None
            data['experiment'] = Experiment(
                **data['experiment'],
                screen=screen,
                eyetracker=eyetracker,
            )

        def reverse_substitute_types(d: Any) -> Any:
            if isinstance(d, dict):
                return {k: reverse_substitute_types(v) for k, v in d.items()}
            if isinstance(d, list):
                return [reverse_substitute_types(v) for v in d]
            if isinstance(d, str) and d.startswith('!'):
                type_name = d[1:]
                if '.' in type_name:
                    module_name, class_name = type_name.rsplit('.', 1)
                    module = importlib.import_module(module_name)
                    return getattr(module, class_name)
                return getattr(builtins, type_name)
            return d

        data = reverse_substitute_types(data)
        # Initialize DatasetDefinition with YAML data
        return DatasetDefinition(**data)

    def to_yaml(self, path: str | Path) -> None:
        """Save a dataset definition to a YAML file.

        Parameters
        ----------
        path: str | Path
            Path where to save the YAML file to.
        """
        data = asdict(self)

        def substitute_types(d: Any) -> Any:
            if isinstance(d, dict):
                return {k: substitute_types(v) for k, v in d.items()}
            if isinstance(d, list):
                return [substitute_types(v) for v in d]
            if isinstance(d, type):
                if d.__module__ == 'builtins':
                    return f'!{d.__name__}'
                return f'!{d.__module__}.{d.__name__}'
            return d

        data['experiment'] = data['experiment'].to_dict()

        data = substitute_types(data)

        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, sort_keys=False)
