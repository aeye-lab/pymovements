Gaze
====

Gaze page

.. currentmodule:: pymovements

.. rubric:: Classes

.. autosummary::
   :toctree: api
   :template: class.rst

    Gaze
    GazeDataFrame

.. currentmodule:: pymovements.gaze.transforms

.. rubric:: Transformations

.. autosummary::

   center_origin
   downsample
   norm
   pix2deg
   deg2pix
   pos2acc
   pos2vel
   savitzky_golay

.. currentmodule:: pymovements.gaze

.. rubric:: Input / Output

.. autosummary::

    from_asc
    from_csv
    from_ipc

.. rubric:: Integration

.. autosummary::

    from_numpy
    from_pandas

.. currentmodule:: pymovements.gaze.transforms_numpy

.. rubric:: Numpy Transformations

.. autosummary::

    pix2deg
    pos2acc
    pos2vel
    norm
    split
    downsample
    consecutive

.. currentmodule:: pymovements

.. toctree::
    :hidden:
    :maxdepth: 2

    transforms/index
    io/index
    integration/index
    transforms_numpy/index
