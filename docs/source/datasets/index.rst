==========
 Datasets
==========

Download your dataset and load it into memory with these few lines of code:

.. code-block:: python

    import pymovements as pm

    # Initialize the dataset object with its name
    # Specify your local directory for saving and loading data
    dataset = pm.Dataset(name='EMTeC', path='path/to/your/data/directory')

    # Download the dataset and extract all archives.
    dataset.download()

    # Load the dataset into memory for processing
    dataset.load()

.. csv-table:: Public Datasets
   :file: public_datasets.csv
   :widths: 30, 35, 5, 5, 5, 5, 5, 5, 5
   :header-rows: 1

.. csv-table:: Example Datasets
   :file: example_datasets.csv
   :widths: 30, 35, 5, 5, 5, 5, 5, 5, 5
   :header-rows: 1
