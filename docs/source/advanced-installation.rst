===========================
Advanced Installation Guide
===========================

Adding conda-forge to your conda repositories:
##############################################

pymovements can be installed from the conda-forge repositories.
If not already done, you will need to add conda-forge to your available channels:

.. code-block:: bash

    conda config --add channels conda-forge
    conda config --set channel_priority strict

You can then install pymovements into you conda environment:

.. code-block:: bash

    conda install -c conda-forge pymovements



Development installation
########################

To use the latest development version or to try out tutorials, pymovements may be alternatively
cloned and installed with

.. code-block:: bash

    git clone https://github.com/aeye-lab/pymovements.git
    pip install --upgrade pip
    pip install -e ./pymovements
