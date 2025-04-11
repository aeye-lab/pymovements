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
"""Benchmark import times."""
import subprocess
import sys


def setup_import_pymovements():
    teardown_import_pymovements()
    assert 'pymovements' not in sys.modules
    assert not any(module.startswith('pymovements') for module in sys.modules)


def import_pymovements():
    import pymovements  # noqa: F401 # pylint: disable=import-outside-toplevel,unused-import


def teardown_import_pymovements():
    try:
        del pymovements  # noqa: F821 # pylint: disable=undefined-variable
    except BaseException:
        pass

    for module in list(sys.modules.keys()):
        if module.startswith('pymovements'):
            del sys.modules[module]


def test_import_pymovements(benchmark):
    benchmark.pedantic(
        import_pymovements,
        setup=setup_import_pymovements,
        iterations=1, rounds=10,
    )


def import_pymovements_subprocess():
    cmd = [sys.executable, '-c', 'import pymovements']
    subprocess.run(cmd, check=True)


def test_import_pymovements_subprocess(benchmark):
    benchmark.pedantic(
        import_pymovements_subprocess,
        iterations=1, rounds=10,
    )


def import_pymovements_subprocess_x():
    cmd = [sys.executable, '-X', 'importtime', '-c', 'import pymovements']
    p = subprocess.run(cmd, stderr=subprocess.PIPE, check=True)

    lines = p.stderr.splitlines()
    line = lines[-1]
    field = line.split(b'|')[-2].strip()
    total = int(field)  # microseconds
    return total, lines


def test_import_pymovements_subprocess_x(benchmark):
    benchmark.pedantic(
        import_pymovements_subprocess_x,
        iterations=1, rounds=10,
    )
