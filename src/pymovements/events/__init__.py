# Copyright (c) 2022-2023 The pymovements Project Authors
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
"""This module holds all event related functionality.

.. rubric:: Classes

.. autosummary::
   :toctree:
   :recursive:
   :template: class.rst

    pymovements.events.EventDataFrame

.. rubric:: Processing

.. autosummary::
   :toctree:
   :recursive:

    pymovements.events.EventProcessor
    pymovements.events.EventGazeProcessor

.. rubric:: Detection Methods

.. autosummary::
   :toctree:
   :recursive:

    pymovements.events.idt
    pymovements.events.ivt
    pymovements.events.microsaccades
    pymovements.events.fill

.. rubric:: Event Properties

.. autosummary::
    :toctree:
    :recursive:

    pymovements.events.event_properties.amplitude
    pymovements.events.event_properties.duration
    pymovements.events.event_properties.dispersion
    pymovements.events.event_properties.disposition
    pymovements.events.event_properties.peak_velocity
    pymovements.events.event_properties.location
"""
from pymovements.events.detection.fill import fill
from pymovements.events.detection.idt import idt
from pymovements.events.detection.ivt import ivt
from pymovements.events.detection.microsaccades import microsaccades
from pymovements.events.event_processing import EventGazeProcessor
from pymovements.events.event_processing import EventProcessor
from pymovements.events.events import EventDataFrame
from pymovements.events.events import register_event_detection

__all__ = [
    'EventDataFrame',
    'EventGazeProcessor',
    'EventProcessor',
    'fill',
    'idt',
    'ivt',
    'microsaccades',
    'register_event_detection',
]
