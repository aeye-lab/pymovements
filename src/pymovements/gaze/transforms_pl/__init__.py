# Copyright (c) 2023 The pymovements Project Authors
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
"""Transforms module."""
from pymovements.gaze.transforms_pl.center_origin import center_origin
from pymovements.gaze.transforms_pl.downsample import downsample
from pymovements.gaze.transforms_pl.norm import norm
from pymovements.gaze.transforms_pl.pix2deg import helper as pix2deg_helper
from pymovements.gaze.transforms_pl.pix2deg import pix2deg
from pymovements.gaze.transforms_pl.pos2acc import pos2acc
from pymovements.gaze.transforms_pl.pos2vel import pos2vel
from pymovements.gaze.transforms_pl.savitzky_golay import helper as savitzky_golay_helper
from pymovements.gaze.transforms_pl.savitzky_golay import savitzky_golay
from pymovements.gaze.transforms_pl.transforms_library import register_transform
from pymovements.gaze.transforms_pl.transforms_library import TransformLibrary


__all__ = [
    'center_origin',
    'downsample',
    'norm',
    'pix2deg',
    'pix2deg_helper',
    'pos2acc',
    'pos2vel',
    'register_transform',
    'savitzky_golay',
    'savitzky_golay_helper',
    'TransformLibrary',
]
