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

    pymovements.events.amplitude
    pymovements.events.duration
    pymovements.events.dispersion
    pymovements.events.disposition
    pymovements.events.peak_velocity
    pymovements.events.location
"""
from pymovements.events.detection import fill
from pymovements.events.detection import idt
from pymovements.events.detection import ivt
from pymovements.events.detection import microsaccades
from pymovements.events.detection._library import EventDetectionLibrary
from pymovements.events.detection._library import register_event_detection
from pymovements.events.frame import EventDataFrame
from pymovements.events.processing import EventGazeProcessor
from pymovements.events.processing import EventProcessor
from pymovements.events.properties import amplitude
from pymovements.events.properties import dispersion
from pymovements.events.properties import disposition
from pymovements.events.properties import duration
from pymovements.events.properties import EVENT_PROPERTIES
from pymovements.events.properties import location
from pymovements.events.properties import peak_velocity
from pymovements.events.properties import register_event_property


__all__ = [
    'EventDetectionLibrary',
    'register_event_detection',
    'fill',
    'idt',
    'ivt',
    'microsaccades',

    'EventDataFrame',

    'EventGazeProcessor',
    'EventProcessor',

    'EVENT_PROPERTIES',
    'register_event_property',
    'amplitude',
    'dispersion',
    'disposition',
    'duration',
    'location',
    'peak_velocity',
]
