import pytest


def test_dataset_fixture_is_dataset(dataset):
    assert isinstance(dataset, pm.Dataset)