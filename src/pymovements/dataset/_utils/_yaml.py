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
"""Yaml utilities."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


# generalized constructor for !* tags
def type_constructor(
        loader: yaml.SafeLoader,
        prefix: str,
        node: yaml.Node,
) -> type:
    """Resolve a YAML tag to a corresponding Python type.

    This function is used to handle custom YAML tags (e.g., `!polars.Int64`)
    by mapping the tag to a Python type or class name. The type name is
    extracted from the YAML tag and evaluated to return the corresponding
    Python object. If the type cannot be resolved, an error is raised.

    Parameters
    ----------
    loader: yaml.SafeLoader
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
        # !polars.Int64
        #
        # Resolves to the Python type `pl.Int64` (assuming `pl` is a valid module).

    """
    # pylint: disable=unused-argument
    # extract the type name (e.g., from !polars.Int64 to polars.Int64)
    type_name = node.tag[1:]

    built_in_types = {
        'int': int,
        'float': float,
        'str': str,
        'bool': bool,
        'list': list,
        'dict': dict,
        'set': set,
    }

    # check for built-in types first
    if type_name in built_in_types:
        return built_in_types[type_name]

    try:
        module_name, type_attr = type_name.rsplit('.', 1)
        module = __import__(module_name)
        return getattr(module, type_attr)

    # module does not have this file type
    except AttributeError as exc:
        raise ValueError(
            f'Unknown type: {type_attr} for module {module_name}',
        ) from exc
    except ValueError as exc:
        raise ValueError(f'Unknown {node=}') from exc


def tuple_constructor(loader: yaml.SafeLoader, node: yaml.Node) -> Any:
    """Construct a Python tuple from a YAML sequence node.

    This function is used as a custom constructor for PyYAML to convert a YAML
    sequence (e.g., a list-like structure) into a Python tuple. It can be registered
    with a YAML loader to handle custom tags like '!tuple'.

    Parameters
    ----------
    loader: yaml.SafeLoader
        The PyYAML loader instance being used to parse the YAML.
    node: yaml.Node
        The YAML node representing the sequence to be converted into a tuple.

    Returns
    -------
    Any
        A Python tuple containing the elements of the YAML sequence.

    Example
    -------
        pixel_columns: !tuple
        - x
        - y
    """
    if not isinstance(node, yaml.SequenceNode):
        raise yaml.YAMLError(f'Expected a SequenceNode, got {type(node)}')

    return tuple(loader.construct_sequence(node))


def tuple_representer(dumper: yaml.Dumper, node: tuple[Any, ...]) -> yaml.Node:
    """Represent a Python tuple as a YAML sequence with a custom '!tuple' tag.

    This function is used as a custom representer for PyYAML to serialize a Python
    tuple into a YAML sequence with the '!tuple' tag. It can be registered with a
    YAML dumper to ensure tuples are represented distinctly from lists.

    Parameters
    ----------
    dumper: yaml.Dumper
        The PyYAML dumper instance being used to serialize the data.
    node: tuple[Any, ...]
        The Python tuple to be represented in YAML.

    Returns
    -------
    yaml.Node
        A YAML sequence node tagged with '!tuple' containing the tuple's elements.

    Example
    -------
        {'pixel_columns': ('x', 'y')}
        pixel_columns: !tuple [x, y]
        OR
        pixel_columns: !tuple
        - x
        - y
    """
    return dumper.represent_sequence('!tuple', node)


def write_dataset_definitions_yaml(
        datasets_yaml_path: str = 'src/pymovements/datasets/datasets.yaml',
) -> None:
    """Automatically write `datasets.yaml` file for registering datasets.

    Parameters
    ----------
    datasets_yaml_path: str
        Where to write the datasets definition.
        (default: src/pymovements/datasets/datasets.yaml)

    """
    dataset_definition_files = Path('src/pymovements/datasets/')
    datasets_list = []

    for yaml_file in dataset_definition_files.iterdir():
        # https://github.com/aeye-lab/pymovements/pull/952#issuecomment-2690742187
        assert isinstance(yaml_file, Path)
        if yaml_file.suffix == '.yaml':
            yaml_filename = yaml_file.parts[-1]
            if yaml_filename == 'datasets.yaml':
                continue
            datasets_list.append(yaml_filename.split('.')[0])

    with open(datasets_yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(sorted(datasets_list), f)
