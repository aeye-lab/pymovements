import shutil

import pytest

from pymovements.datasets import JuDo1000

@pytest.fixture()
def judo1000_dataset():
    #dataset = JuDo1000(
    #    root='data/',
    #    download=True,
    #)
    yield None

    # remove dataset after all tests
    #shutil.rmtree(dataset.dirpath)


class TestJuDo1000:
    def test_judo1000(self, judo1000_dataset):
        assert judo1000_dataset is None
