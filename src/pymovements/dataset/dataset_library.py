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

from copy import deepcopy
from importlib import resources
from pathlib import Path
from typing import TypeVar

from pymovements import datasets
from pymovements.dataset.dataset_definition import DatasetDefinition


class DatasetLibrary:
    """Provides access by name to :py:class:`~pymovements.dataset.DatasetDefinition`.

    Attributes
    ----------
    definitions: dict[str, DatasetDefinition]
        Dictionary of :py:class:`~pymovements.dataset.DatasetDefinition`,
        either as classes or instances.
    """

    definitions: dict[str, DatasetDefinition] = {}

    @classmethod
    def add(cls, definition: type[DatasetDefinition] | Path | str) -> None:
        """Add :py:class:`~pymovements.dataset.DatasetDefinition` to library.

        Parameters
        ----------
        definition: type[DatasetDefinition] | Path | str
            The :py:class:`~pymovements.dataset.DatasetDefinition` to add to the library.

        Notes
        -----
        Definition can be:
            * A DatasetDefinition class
            * A DatasetDefinition instance
            * A Path to a YAML file
            * A string path to a YAML file
        """
        if isinstance(definition, (str, Path)):
            # Load from YAML file
            yaml_def = DatasetDefinition.from_yaml(definition)
            cls.definitions[yaml_def.name] = yaml_def
        else:
            cls.definitions[definition.name] = definition()

    @classmethod
    def get(cls, name: str) -> DatasetDefinition:
        """Get :py:class:`~pymovements.dataset.DatasetDefinition` py name.

        Parameters
        ----------
        name: str
            Name of the dataset definition in the library.

        Returns
        -------
        DatasetDefinition
            The :py:class:`~pymovements.dataset.DatasetDefinition`.

        Raises
        ------
        KeyError
            If dataset name not found in library.
        """
        if name not in cls.definitions:
            raise KeyError(
                f"Dataset '{name}' not found in DatasetLibrary. "
                f"Available datasets: {sorted(cls.definitions.keys())}",
            )
        return deepcopy(cls.definitions[name])

    @classmethod
    def names(cls) -> list[str]:
        """Return available datasets in :py:class:`~pymovements.dataset.DatasetLibrary`.

        Returns
        -------
        list[str]
            List of dataset names that are available in
            :py:class:`~pymovements.dataset.DatasetLibrary`.
        """
        return sorted(list(cls.definitions.keys()))


DatasetDefinitionClass = TypeVar('DatasetDefinitionClass', bound=type[DatasetDefinition])


def register_dataset(cls: DatasetDefinitionClass) -> DatasetDefinitionClass:
    """Register a public dataset definition.

    Parameters
    ----------
    cls: DatasetDefinitionClass
        The :py:class:`~pymovements.dataset.DatasetDefinition` to register.

    Returns
    -------
    DatasetDefinitionClass
        The :py:class:`~pymovements.dataset.DatasetDefinition` to register.
    """
    DatasetLibrary.add(cls)
    return cls


def _add_shipped_datasets() -> None:
    """Add available public datasets via yaml files in `src/pymovements/datasets`."""
    dataset_definition_files = resources.files(datasets)

    for filename in dataset_definition_files.iterdir():
        # mypy... don't know how to make it smarter currently (type is Traversable)
        assert isinstance(filename, Path)
        dataset_path = Path(filename)
        if (
            dataset_path.name == 'datasets.yaml' or
            dataset_path.suffix == '.py' or
            dataset_path.is_dir()
        ):
            continue
        # https://github.com/aeye-lab/pymovements/pull/952#issuecomment-2690742187
        assert isinstance(dataset_definition_files, Path)
        yaml_file_name = dataset_definition_files / filename
        # https://github.com/aeye-lab/pymovements/pull/952#issuecomment-2690742187
        assert isinstance(yaml_file_name, Path)
        DatasetLibrary.add(yaml_file_name)


_add_shipped_datasets()
