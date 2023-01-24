import pytest

from pymovements.datasets import PublicDataset


def test_no_mirrors_download_exception():
    class NoMirrorsPublicDataset(PublicDataset):
        mirrors = []

    with pytest.raises(ValueError):
        dataset = NoMirrorsPublicDataset(
            root='data/',
            download=True,
        )


def test_no_resources_download_exception():
    class NoResourcesPublicDataset(PublicDataset):
        mirrors = ['https://mirror.url']
        resources = []

    with pytest.raises(ValueError):
        dataset = NoResourcesPublicDataset(
            root='data/',
            download=True,
        )
