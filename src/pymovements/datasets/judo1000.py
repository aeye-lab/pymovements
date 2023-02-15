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
"""This module provides an interface to the JuDo1000 dataset."""
from pymovements.base import Experiment
from pymovements.datasets.base import PublicDataset


class JuDo1000(PublicDataset):
    """JuDo1000 dataset :cite:p:`JuDo1000`.

    This dataset includes binocular eye tracking data from 150 participants in four sessions with an
    interval of at least one week between two sessions. Eye movements are recorded at a sampling
    frequency of 1000 Hz using an EyeLink Portable Duo video-based eye tracker and are provided as
    pixel coordinates. Participants are instructed to watch a random jumping dot on a computer
    screen.

    Check the respective `repository <'https://osf.io/download/4wy7s/'>` for details.
    """
    _mirrors = [
        'https://osf.io/download/',
    ]

    _resources = [
        {
            'resource': '4wy7s/',
            'filename': 'JuDo1000.zip',
            'md5': 'b8b9e5bb65b78d6f2bd260451cdd89f8',
        },
    ]

    _experiment = Experiment(
        screen_width_px=1280,
        screen_height_px=1024,
        screen_width_cm=38,
        screen_height_cm=30.2,
        distance_cm=68,
        origin='lower left',
        sampling_rate=1000,
    )

    _filename_regex = r'(?P<subject_id>\d+)_(?P<session_id>\d+).csv'

    _filename_regex_dtypes = {
        'subject_id': int,
        'session_id': int,
    }

    _column_map = {
        'trialId': 'trial_id',
        'pointId': 'point_id',
        'time': 'time',
        'x_left': 'x_left_pix',
        'y_left': 'y_left_pix',
        'x_right': 'x_right_pix',
        'y_right': 'y_right_pix',
    }

    _read_csv_kwargs = {
        'sep': '\t',
        'columns': list(_column_map.keys()),
        'new_columns': list(_column_map.values()),
    }

    def __init__(self, **kwargs):
        super().__init__(
            experiment=self._experiment,
            filename_regex=self._filename_regex,
            filename_regex_dtypes=self._filename_regex_dtypes,
            custom_read_kwargs=self._read_csv_kwargs,
            **kwargs,
        )
