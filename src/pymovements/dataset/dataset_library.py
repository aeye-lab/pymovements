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
"""DatasetLibrary module."""
from __future__ import annotations

from importlib import resources
from pathlib import Path
from pathlib import PosixPath

import yaml

from pymovements import datasets
from pymovements.dataset.dataset_definition import DatasetDefinition


class DatasetLibrary:
    """Provides access by name to dataset definitions.

    Attributes
    ----------
    definitions: dict[str, DatasetDefinition]
        Dictionary of :py:class:`~pymovements.DatasetDefinition`,
        either as classes or instances.
    """

    definitions: dict[str, DatasetDefinition] = {}

    @classmethod
    def add(cls, definition: DatasetDefinition | Path | str) -> None:
        """Add a dataset definition to library.

        Parameters
        ----------
        definition: DatasetDefinition | Path | str
            The :py:class:`~pymovements.DatasetDefinition` to add.

        Notes
        -----
        Definition can be:
            * A DatasetDefinition class (legacy)
            * A DatasetDefinition instance (from YAML)
            * A Path to a YAML file
            * A string path to a YAML file
        """
        if isinstance(definition, (str, Path)):
            # Load from YAML file
            yaml_def = DatasetDefinition.from_yaml(definition)
            cls.definitions[yaml_def.name] = yaml_def
        else:
            # DatasetDefinition instance
            cls.definitions[definition.name] = definition

    @classmethod
    def get(cls, name: str) -> DatasetDefinition:
        """Get dataset definition by name.

        Parameters
        ----------
        name: str
            Name of the dataset definition in the library.

        Returns
        -------
        DatasetDefinition
            The :py:class:`~pymovements.DatasetDefinition`.
            Could be either a class (legacy) or instance (YAML).

        Raises
        ------
        KeyError
            If dataset name not found in library.
        """
        if name not in cls.definitions:
            raise KeyError(
                f"Dataset '{name}' not found in library. "
                f"Available datasets: {sorted(cls.definitions.keys())}",
            )
        return cls.definitions[name]

    @classmethod
    def register_yaml_directory(cls, directory: str | Path) -> None:
        """Register all YAML dataset definitions in a directory.

        Parameters
        ----------
        directory: str | Path
            Directory containing YAML dataset definitions
        """
        directory = Path(directory)
        for yaml_file in directory.glob('*.yaml'):
            if yaml_file.parts[-1] == 'datasets.yaml':
                continue
            cls.add(yaml_file)


def register_dataset(cls: DatasetDefinition) -> DatasetDefinition:
    """Register a public dataset definition.

    Parameters
    ----------
    cls: DatasetDefinition
        The :py:class:`~pymovements.DatasetDefinition` to register.

    Returns
    -------
    DatasetDefinition
        The :py:class:`~pymovements.DatasetDefinition` to register.
    """
    DatasetLibrary.add(cls)
    return cls


def _add_shipped_datasets() -> None:
    """Add available public datasets via `src/pymovements/datasets/datasets.yaml`."""
    dataset_definition_files = resources.files(datasets)

    datasets_list_yaml = dataset_definition_files / 'datasets.yaml'
    assert isinstance(datasets_list_yaml, PosixPath)
    with open(datasets_list_yaml, encoding='utf-8') as f:
        datasets_list = yaml.safe_load(f)

    for definition_basename in datasets_list:
        yaml_file_name = dataset_definition_files / f'{definition_basename}.yaml'
        assert isinstance(yaml_file_name, PosixPath)
        DatasetLibrary.add(yaml_file_name)


_add_shipped_datasets()
