# Copyright (c) 2025 The pymovements Project Authors
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
"""YAMLDatasetLoader class."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import yaml

from pymovements.dataset.dataset_definition import DatasetDefinition
from pymovements.gaze.experiment import Experiment


# generalized constructor for !* tags
def type_constructor(
        loader: yaml.Loader | yaml.FullLoader | yaml.UnsafeLoader,
        prefix: str,
        node: yaml.Node,
) -> type:
    """Resolve a YAML tag to a corresponding Python type.

    This function is used to handle custom YAML tags (e.g., `!pl.Int64`)
    by mapping the tag to a Python type or class name. The type name is
    extracted from the YAML tag and evaluated to return the corresponding
    Python object. If the type cannot be resolved, an error is raised.

    Parameters
    ----------
    loader: yaml.Loader | yaml.FullLoader | yaml.UnsafeLoader
        The YAML loader being used to parse the YAML document.
    prefix: str
        A string prefix for the custom tag (e.g., '!').
    node: yaml.Node
        The YAML node containing the tag and associated value.

    Returns
    -------
    type
        The Python type or class corresponding to the YAML tag.

    Raises
    ------
    ValueError: If the specified type name in the tag cannot be resolved
                to a valid Python object.

    Example:
        # Example YAML document:
        # !pl.Int64
        #
        # Resolves to the Python type `pl.Int64` (assuming `pl` is a valid module).

    """
    # pylint: disable=unused-argument
    # extract the type name (e.g., from !pl.Int64 to pl.Int64)
    type_name = node.tag[1:]

    built_in_types = {
        'int': int,
        'float': float,
        'str': str,
        'bool': bool,
        'list': list,
        'dict': dict,
        'set': set,
        'tuple': tuple,
    }

    # check for built-in types first
    if type_name in built_in_types:
        return built_in_types[type_name]

    try:
        module_name, type_attr = type_name.rsplit('.', 1)
        module = __import__(module_name)
        return getattr(module, type_attr)

    except AttributeError as exc:
        raise ValueError(f"Unknown type: {type_name}") from exc


yaml.add_multi_constructor('!', type_constructor)


class YAMLDatasetLoader:
    """Loads dataset definitions from YAML files."""

    @staticmethod
    def load_dataset_definition(yaml_path: str | Path) -> DatasetDefinition:
        """Load a dataset definition from a YAML file.

        Parameters
        ----------
        yaml_path : str | Path
            Path to the YAML definition file

        Returns
        -------
        DatasetDefinition
            Initialized dataset definition
        """
        with open(yaml_path, encoding='utf-8') as f:
            data = yaml.load(f, Loader=yaml.Loader)

        # Convert experiment dict to Experiment object if present
        if 'experiment' in data:
            data['experiment'] = Experiment(**data['experiment'])

        # Initialize DatasetDefinition with YAML data
        return DatasetDefinition(**data)

    @staticmethod
    def save_dataset_definition(definition: DatasetDefinition, yaml_path: str | Path) -> None:
        """Save a dataset definition to a YAML file.

        Parameters
        ----------
        definition : DatasetDefinition
            Dataset definition to save
        yaml_path : str | Path
            Path where to save the YAML file
        """
        # Convert to dict and handle experiment object
        data = asdict(definition)
        if data['experiment']:
            data['experiment'] = asdict(data['experiment'])

        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, sort_keys=False)
