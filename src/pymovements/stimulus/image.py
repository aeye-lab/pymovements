# Copyright (c) 2023-2025 The pymovements Project Authors
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
"""Module for the ImageDataFrame."""
from __future__ import annotations

from pathlib import Path

from pymovements.utils.paths import get_filepaths
from pymovements.utils.plotting import draw_image_stimulus
from pymovements.utils.strings import curly_to_regex


class ImageStimulus:
    """A DataFrame for image stimulus.

    Parameters
    ----------
    images: list[Path]
        Image stimulus list.
    """

    def __init__(self, images: list[Path]) -> None:
        self.images = images

    def show(self, stimulus_id: int, origin: str = 'upper') -> None:
        """Show image stimulus.

        Parameters
        ----------
        stimulus_id: int
            Number of stimulus to be shown.
        origin: str
            Origin of the stimulus to be shown.
        """
        draw_image_stimulus(self.images[stimulus_id], origin=origin, show=True)


def from_file(image_path: str | Path) -> ImageStimulus:
    """Load image stimulus from file.

    Parameters
    ----------
    image_path:  str | Path
        Path to file to be read.

    Returns
    -------
    ImageStimulus
        Returns the image stimulus file.
    """
    if isinstance(image_path, str):
        image_path = Path(image_path)

    return ImageStimulus(images=[image_path])


def from_files(path: str | Path, filename_format: str) -> ImageStimulus:
    """Load image stimulus from file.

    Parameters
    ----------
    path:  str | Path
        Path to directory with image stimulus files.
    filename_format:  str
        Format of the image stimulus file names.

    Returns
    -------
    ImageStimulus
        Returns the image stimulus file.
    """
    filenames = get_filepaths(
        path,
        regex=curly_to_regex(filename_format),
    )
    image_stimuli = []
    for filename in filenames:
        image_stimuli.append(filename)

    return ImageStimulus(image_stimuli)
