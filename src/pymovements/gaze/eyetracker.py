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
"""This module holds the Screen class."""
from __future__ import annotations


from pymovements.utils import decorators


@decorators.auto_str
class EyeTracker:
    """EyeTracker class for holding eyetracker properties.

    Attributes
    ----------
    sampling_rate_hz : int
        Sample rate of recording (in Hz)
    eyes : set[str]
        Set of tracked eyes
    model : str
        EyeLink tracker model (e.g. 'EyeLink II', 'EyeLink 1000')
    version : str
        EyeLink software version number
    mount : str
        The mounting setup of the EyeLink (e.g. 'Desktop / Monocular / Remote')

    """

    def __init__(
            self,
            sampling_rate_hz: int,
            eyes: set[str],
            model: str,
            version: str,
            mount: str,
    ):
        """Initialize Eyetracker.

        Parameters
        ----------
        sampling_rate_hz : int
            Sample rate of recording (in Hz)
        eyes : set[str]
            Set of tracked eyes
        model : str
            EyeLink tracker model (e.g. 'EyeLink II', 'EyeLink 1000')
        version : str
            EyeLink software version number
        mount : str
            The mounting setup of the EyeLink (e.g. 'Desktop / Monocular / Remote')

        Examples
        --------
        >>> eyetracker = EyeTracker(
        ...     sampling_rate_hz = 1000,
        ...     eyes = {'right'},
        ...     model = 'EyeLink 1000 Plus',
        ...     version = '1.5.3',
        ...     mount = 'Arm Mount / Monocular / Remote',
        ... )
        >>> print(eyetracker)

        """
        self.sampling_rate_hz = sampling_rate_hz
        self.eyes = eyes
        self.model = model
        self.version = version
        self.mount = mount
