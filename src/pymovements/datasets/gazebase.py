"""This module provides an interface to the GazeBase dataset."""
from pymovements.base import Experiment
from pymovements.datasets.base import PublicDataset


class GazeBase(PublicDataset):
    """GazeBase dataset :cite:p:`GazeBase`.

    Check the respective `repository
    <'https://figshare.com/articles/dataset/GazeBase_Data_Repository/12912257'>` for details.
    """
    _mirrors = [
        'https://figshare.com/ndownloader/files/',
    ]

    _resources = [
        {
            'resource': '27039812',
            'filename': "GazeBase_v2_0.zip",
            'md5': 'cb7eb895fb48f8661decf038ab998c9a',
        },
    ]

    # FIXME: we don't have any of these screen informations and gaze data is already given in dva
    _experiment = Experiment(
        screen_width_px=1280,
        screen_height_px=1024,
        screen_width_cm=38,
        screen_height_cm=30.2,
        distance_cm=68,
        sampling_rate=1000,
    )

    _filename_regex = (
        r'S_(?P<round_id>\d)(?P<subject_id>\d+)'
        r'_S(?P<session_id>\d+)'
        r'_(?P<task_name>.+).csv'
    )

    _filename_regex_dtypes = {
        'round_id': int,
        'subject_id': int,
        'session_id': int,
    }

    _column_map = {
        'n': 'time',
        'x': 'x_left_dva',
        'y': 'y_left_dva',
        'val': 'val',
        'xT': 'x_target_dva',
        'yT': 'y_target_dva',
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
