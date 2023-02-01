import re

import pandas as pd

from pymovements.base import Experiment
from pymovements.datasets import PublicDataset


class JuDo1000(PublicDataset):
    """JuDo1000 dataset.

    :cite:p:`JuDo1000`

    """
    mirrors = [
        'https://osf.io/download/',
    ]

    resources = [
        {'path': '4wy7s/', 'filename': "JuDo1000.zip", 'md5': 'b8b9e5bb65b78d6f2bd260451cdd89f8'},
    ]

    experiment = Experiment(
        screen_width_px=1280,
        screen_height_px=1024,
        screen_width_cm=38,
        screen_height_cm=30.2,
        distance_cm=68,
        sampling_rate=1000,
    )

    filename_regex = re.compile(
        r'(?P<subject_id>\d+)'
        r'_(?P<session_id>\d+).csv'
    )

    filename_regex_dtypes = {
        'subject_id': int,
        'session_id': int,
    }

    column_map = {
        'trialId': 'trial_id',
        'pointId': 'point_id',
        'time': 'time',
        'x_left': 'x_left_pixel',
        'y_left': 'y_left_pixel',
        'x_right': 'x_right_pixel',
        'y_right': 'y_right_pixel',
    }

    read_csv_kwargs = {
        'sep': '\t',
        'columns': list(column_map.keys()),
        'new_columns': list(column_map.values()),
    }

    def __init__(self, **kwargs):
        super().__init__(custom_csv_kwargs=self.read_csv_kwargs, **kwargs)
