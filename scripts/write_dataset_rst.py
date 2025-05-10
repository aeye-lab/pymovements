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
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

import pymovements as pm


def main(
        dataset_name: str,
        datasets_dirpath: str | Path = 'docs/source/datasets',
        datasets_yaml_filename: str = 'datasets.yml',
        datasets_definition_dirname: str = 'definitions',
) -> int:
    if isinstance(datasets_dirpath, str):
        datasets_dirpath = Path(datasets_dirpath)

    dataset_names = [definition.name for definition in pm.DatasetLibrary.definitions.values()]

    with open(datasets_dirpath / datasets_yaml_filename, 'w') as f:
        yaml.dump(dataset_names, f)

    definition = pm.DatasetLibrary.get(dataset_name)

    definition.to_yaml(datasets_dirpath / datasets_definition_dirname / f'{definition.name}.yml')

    rst_content = f'''.. datatemplate:yaml:: definitions/{definition.name}.yml
   :template: dataset.rst
'''

    with open(datasets_dirpath / f'{definition.name}.rst', 'w') as rst_file:
        rst_file.write(rst_content)

    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset')
    args = parser.parse_args()
    raise SystemExit(main(dataset_name=args.dataset))
