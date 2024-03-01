# Copyright (c) 2023-2024 The pymovements Project Authors
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
"""Tests pymovements asc to csv processing."""
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

import pymovements as pm

ASC_TEXT = r"""
** DATE: Wed Mar  8 09:25:20 2023
** TYPE: EDF_FILE BINARY EVENT SAMPLE TAGGED
** VERSION: EYELINK II 1
** SOURCE: EYELINK CL
** EYELINK II CL v6.12 Feb  1 2018 (EyeLink Portable Duo)
** CAMERA: EyeLink USBCAM Version 1.01
** SERIAL NUMBER: CLU-DAB50
** CAMERA_CONFIG: DAB50200.SCD
** RECORDED BY pymovements
** SREB2.2.299 WIN32 LID:20A87A96 Mod:2023.03.08 11:03 MEZ
**

some
lines
to
ignore
MSG	421491 DISPLAY_COORDS 0 0 1679 1049
MSG	421491 RETRACE_INTERVAL  16.5783460286
PRESCALER	1
VPRESCALER	1
PUPIL	AREA
EVENTS	GAZE	LEFT	RATE	1000.00	TRACKING	CR	FILTER	2
SAMPLES	GAZE	LEFT	RATE	1000.00	TRACKING	CR	FILTER	2	INPUT
the next line has all additional trial columns set to None
10000000	  850.7	  717.5	  714.0	    0.0	...
MSG 10000001 START_A
the next line now should have the task column set to A
10000002	  850.7	  717.5	  714.0	    0.0	...
MSG 10000003 STOP_A
the task should be set to None again
10000004	  850.7	  717.5	  714.0	    0.0	...
MSG 10000005 START_B
the next line now should have the task column set to B
10000006	  850.7	  717.5	  714.0	    0.0	...
MSG 10000007 START_TRIAL_1
the next line now should have the trial column set to 1
10000008	  850.7	  717.5	  714.0	    0.0	...
MSG 10000009 STOP_TRIAL_1
MSG 10000010 START_TRIAL_2
the next line now should have the trial column set to 2
10000011	  850.7	  717.5	  714.0	    0.0	...
MSG 10000012 STOP_TRIAL_2
MSG 10000013 START_TRIAL_3
the next line now should have the trial column set to 3
10000014	  850.7	  717.5	  714.0	    0.0	...
MSG 10000015 STOP_TRIAL_3
MSG 10000016 STOP_B
task and trial should be set to None again
10000017	  850.7	  717.5	  .	    0.0	...
"""

PATTERNS = [
    {
        'pattern': 'START_A',
        'column': 'task',
        'value': 'A',
    },
    {
        'pattern': 'START_B',
        'column': 'task',
        'value': 'B',
    },
    {
        'pattern': ('STOP_A', 'STOP_B'),
        'column': 'task',
        'value': None,
    },

    r'START_TRIAL_(?P<trial_id>\d+)',
    {
        'pattern': r'STOP_TRIAL',
        'column': 'trial_id',
        'value': None,
    },
]

EXPECTED_DF = pl.from_dict(
    {
        'time': [10000000, 10000002, 10000004, 10000006, 10000008, 10000011, 10000014, 10000017],
        'x_pix': [850.7, 850.7, 850.7, 850.7, 850.7, 850.7, 850.7, 850.7],
        'y_pix': [717.5, 717.5, 717.5, 717.5, 717.5, 717.5, 717.5, 717.5],
        'pupil': [714.0, 714.0, 714.0, 714.0, 714.0, 714.0, 714.0, np.nan],
        'task': [None, 'A', None, 'B', 'B', 'B', 'B', None],
        'trial_id': [None, None, None, None, '1', '2', '3', None],
    },
)

EXPECTED_METADATA = {
    'weekday': 'Wed',
    'month': 'Mar',
    'day': 8,
    'time': '09:25:20',
    'year': 2023,
    'version_1': 'EYELINK II 1',
    'version_2': 'EYELINK II CL v6.12 Feb  1 2018 (EyeLink Portable Duo)',
    'model': 'EyeLink Portable Duo',
    'version_number': 6.12,
    'sampling_rate': 1000.00,
    'filter': '2',
    'tracking': 'CR',
    'tracked_eye': 'LEFT',
    'calibrations': [],
    'validations': [],
    'resolution': (0, 0, 1679, 1049),
}


def test_parse_eyelink(tmp_path):
    filepath = tmp_path / 'sub.asc'
    filepath.write_text(ASC_TEXT)

    df, metadata = pm.utils.parsing.parse_eyelink(
        filepath,
        patterns=PATTERNS,
    )

    assert_frame_equal(df, EXPECTED_DF, check_column_order=False)
    assert metadata == EXPECTED_METADATA


@pytest.mark.parametrize(
    'patterns',
    [
        [1],
        [{'pattern': 1}],
    ],
)
def test_parse_eyelink_raises_value_error(tmp_path, patterns):
    filepath = tmp_path / 'sub.asc'
    filepath.write_text(ASC_TEXT)

    with pytest.raises(ValueError) as excinfo:
        pm.utils.parsing.parse_eyelink(
            filepath,
            patterns=patterns,
        )

    msg, = excinfo.value.args

    expected_substrings = ['invalid pattern', '1']
    for substring in expected_substrings:
        assert substring in msg


@pytest.mark.parametrize(
    'metadata, expected_version, expected_model, time',
    [
        pytest.param(
            '** DATE: Wed Mar  8 09:25:20 2023\n'
            '** VERSION: EYELINK II 1\n'
            '** EYELINK II CL v6.12 Feb  1 2018 (EyeLink Portable Duo)',
            6.12,
            'EyeLink Portable Duo',
            '09:25:20',
            id='eye_link_portable_duo',
        ),
        pytest.param(
            '** DATE: Wed Mar  8 09:25:20 2023\n'
            '** VERSION: EYELINK II 1\n'
            '** EYELINK II CL v5.12 Feb  1 2018',
            5.12,
            'EyeLink 1000 Plus',
            '09:25:20',
            id='eye_link_1000_plus',
        ),
        pytest.param(
            '** DATE: Wed Mar  8 09:25:20 2023\n'
            '** VERSION: EYELINK II 1\n'
            '** EYELINK II CL v4.12 Feb  1 2018',
            4.12,
            'EyeLink 1000',
            '09:25:20',
            id='eye_link_1000_1',
        ),
        pytest.param(
            '** DATE: Wed Mar  8 09:25:20 2023\n'
            '** VERSION: EYELINK II 1\n'
            '** EYELINK II CL v3.12 Feb  1 2018',
            3.12,
            'EyeLink 1000',
            '09:25:20',
            id='eye_link_1000_2',
        ),
        pytest.param(
            '** DATE: Wed Mar  8 09:25:20 2023\n'
            '** VERSION: EYELINK II 1\n'
            '** EYELINK II CL v2.12 Feb  1 2018',
            2.12,
            'EyeLink II',
            '09:25:20',
            id='eye_link_II',
        ),
        pytest.param(
            '** DATE: Wed Mar  8 09:25:20 2023\n'
            '** VERSION: EYELINK REVISION 2.00 (Aug 12 1997)',
            2.00,
            'EyeLink I',
            '09:25:20',
            id='eye_link_I',
        ),
        pytest.param(
            '** DATE: Wed Mar  8 09:25:20 2023\n'
            '** VERSION: nothing\n',
            'unknown',
            'unknown',
            '09:25:20',
            id='unknown_version_1',
        ),
        pytest.param(
            '** DATE: Wed Mar  8 09:25:20 2023\n'
            '** VERSION: EYELINK II 1\n'
            '** EYELINK II CL Feb  1 2018 (EyeLink Portable Duo)',
            'unknown',
            'unknown',
            '09:25:20',
            id='unknown_version_2',
        ),
        pytest.param(
            '** DATE: Wed Mar  8 09:25:20 2023\n'
            '** TYPE: EDF_FILE BINARY EVENT SAMPLE TAGGED',
            'unknown',
            'unknown',
            '09:25:20',
            id='unknown_version_3',
        ),
    ],
)
def test_parse_eyelink_version(tmp_path, metadata, expected_version, expected_model, time):
    filepath = tmp_path / 'sub.asc'
    filepath.write_text(metadata)

    _, metadata = pm.utils.parsing.parse_eyelink(
        filepath,
    )

    assert metadata['version_number'] == expected_version
    assert metadata['model'] == expected_model
    assert metadata['time'] == time


@pytest.mark.parametrize(
    'metadata, expected_msg',
    [
        pytest.param(
            '',
            'No metadata found. Please check the file for errors.',
            id='eye_link_no_metadata',
        ),
    ],
)
def test_no_metadata_warning(tmp_path, metadata, expected_msg):
    filepath = tmp_path / 'sub.asc'
    filepath.write_text(metadata)

    with pytest.raises(Warning) as info:
        _, metadata = pm.utils.parsing.parse_eyelink(
            filepath,
        )

    msg = info.value.args[0]

    assert msg == expected_msg


@pytest.mark.parametrize(
    ('metadata', 'expected_validation', 'expected_calibration'),
    [
        pytest.param(
            '** DATE: Wed Feb  1 04:38:54 2017\n'
            'MSG	7045618 !CAL \n'
            '>>>>>>> CALIBRATION (HV9,P-CR) FOR LEFT: <<<<<<<<<\n'
            'MSG	7045618 !CAL Calibration points:  \n'
            'MSG	1076158 !CAL VALIDATION HV9 R RIGHT POOR ERROR 2.40 avg. 6.03 max  '
            'OFFSET 0.19 deg. 4.2,6.3 pix.\n',
            [{
                'error': 'POOR ERROR',
                'eye_tracked': 'RIGHT',
                'num_points': '9',
                'timestamp': '1076158',
                'validation_score_avg': '2.40',
                'validation_score_max': '6.03',
            }],
            [{
                'num_points': '9',
                'timestamp': '7045618',
                'tracked_eye': 'LEFT',
                'type': 'P-CR',
            }],
            id='cal_timestamp_with_space',
        ),
        pytest.param(
            '** DATE: Wed Feb  1 04:38:54 2017\n'
            'MSG	7045618 !CAL\n'
            '>>>>>>> CALIBRATION (HV9,P-CR) FOR LEFT: <<<<<<<<<\n',
            [],
            [{
                'num_points': '9',
                'timestamp': '7045618',
                'tracked_eye': 'LEFT',
                'type': 'P-CR',
            }],
            id='cal_timestamp_no_space_no_val',
        ),
    ],
)
def test_val_cal_eyelink(tmp_path, metadata, expected_validation, expected_calibration):
    filepath = tmp_path / 'sub.asc'
    filepath.write_text(metadata)

    _, parsed_metadata = pm.utils.parsing.parse_eyelink(filepath)

    assert parsed_metadata['calibrations'] == expected_calibration
    assert parsed_metadata['validations'] == expected_validation


def test_parse_val_cal_eyelink_monocular_file():
    example_asc_monocular_path = Path(__file__).parent.parent.parent / \
        'files/eyelink_monocular_example.asc'

    _, metadata = pm.utils.parsing.parse_eyelink(example_asc_monocular_path)

    expected_validation = [{
        'error': 'GOOD ERROR',
        'eye_tracked': 'LEFT',
        'num_points': '9',
        'timestamp': '2148587',
        'validation_score_avg': '0.27',
        'validation_score_max': '0.83',
    }]
    expected_calibration = [{
        'num_points': '9', 'type': 'P-CR', 'tracked_eye': 'LEFT', 'timestamp': '2135819',
    }]

    assert metadata['calibrations'] == expected_calibration
    assert metadata['validations'] == expected_validation
