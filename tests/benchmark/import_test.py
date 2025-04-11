import subprocess
import sys

import pytest


def setup_import_pymovements():
    teardown_import_pymovements()
    assert 'pymovements' not in sys.modules
    assert not any(module.startswith('pymovements') for module in sys.modules)


def import_pymovements():
    import pymovements


def teardown_import_pymovements():
    try:
        del pymovements
    except:
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
    cmd = [sys.executable, "-c", "import pymovements"]
    subprocess.run(cmd)


def test_import_pymovements_subprocess(benchmark):
    result = benchmark.pedantic(
        import_pymovements_subprocess,
        iterations=1, rounds=10,
    )


def import_pymovements_subprocess_x():
    cmd = [sys.executable, "-X", "importtime", "-c", "import pymovements"]
    p = subprocess.run(cmd, stderr=subprocess.PIPE)

    lines = p.stderr.splitlines()
    line = lines[-1]
    field = line.split(b"|")[-2].strip()
    total = int(field)  # microseconds
    return total, lines


def test_import_pymovements_subprocess_x(benchmark):
    result = benchmark.pedantic(
        import_pymovements_subprocess_x,
        iterations=1, rounds=10,
    )
