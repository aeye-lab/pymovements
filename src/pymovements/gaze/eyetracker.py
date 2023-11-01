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
"""This module holds the EyeTracker class."""
from __future__ import annotations

from dataclasses import dataclass

from pymovements.utils import checks


@dataclass
class EyeTracker:

    """EyeTracker class for holding eyetracker properties.

    Attributes
    ----------
    sampling_rate : float
        Sample rate of recording (in Hz)
    left : bool
        Whether the left eye is tracked
    right : bool
        Whether the right eye is tracked
    model : str
        EyeLink tracker model (e.g. 'EyeLink II', 'EyeLink 1000')
    version : str
        EyeLink software version number
    mount : str
        The mounting setup of the EyeLink (e.g. 'Desktop / Monocular / Remote')

    Examples
        --------
        >>> eyetracker = EyeTracker(
        ...     sampling_rate = 1000.0,
        ...     left = False,
        ...     right = True,
        ...     model = 'EyeLink 1000 Plus',
        ...     version = '1.5.3',
        ...     mount = 'Arm Mount / Monocular / Remote',
        ... )
        >>> print(eyetracker)
        EyeTracker(sampling_rate=1000.00, left=False, right=True, model=EyeLink 1000 Plus,
        version=1.5.3, mount=Arm Mount / Monocular / Remote)
    """
    sampling_rate: float
    left: bool
    right: bool
    model: str = ''
    version: str = ''
    mount: str = ''

    def __post_init__(self):
        """Checks that the sampling rate is a positive number. """
        checks.check_is_not_none(sampling_rate=self.sampling_rate)
        assert self.sampling_rate is not None

        checks.check_is_greater_than_zero(sampling_rate=self.sampling_rate)
