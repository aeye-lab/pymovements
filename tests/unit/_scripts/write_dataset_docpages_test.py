from pymovements import DatasetLibrary
from pymovements._scripts import write_dataset_docpages

def test_write_dataset_docpages(tmp_path):
    write_dataset_docpages.main(doc_dirpath=tmp_path)

    assert (tmp_path / 'datasets.yaml').is_file()

    for dataset_name in DatasetLibrary.names():
        assert (tmp_path / f'{dataset_name}.rst').is_file()
        print(list((tmp_path / 'meta').iterdir()))
        assert (tmp_path / 'meta' / f'{dataset_name}.yaml').is_file()
