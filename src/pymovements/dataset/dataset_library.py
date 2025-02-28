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

from pathlib import Path

from pymovements.dataset.dataset_definition import DatasetDefinition
from pymovements.dataset.yaml_dataset_loader import YAMLDatasetLoader


class DatasetLibrary:
    """Provides access by name to dataset definitions.

    Attributes
    ----------
    definitions: dict[str, DatasetDefinition]
        Dictionary of dataset definitions, either as classes or instances
    """

    definitions: dict[str, DatasetDefinition] = {}

    @classmethod
    def add(cls, definition: DatasetDefinition | Path | str) -> None:
        """Add a dataset definition to library.

        Parameters
        ----------
        definition: DatasetDefinition | Path | str
            The dataset definition to add. Can be:
            - A DatasetDefinition class (legacy)
            - A DatasetDefinition instance (from YAML)
            - A Path to a YAML file
            - A string path to a YAML file
        """
        if isinstance(definition, (str, Path)):
            # Load from YAML file
            yaml_def = YAMLDatasetLoader.load_dataset_definition(definition)
            cls.definitions[yaml_def.name] = yaml_def
        else:
            # DatasetDefinition instance (from YAML)
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
            The dataset definition. Could be either a class (legacy) or instance (YAML).

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
            cls.add(yaml_file)
