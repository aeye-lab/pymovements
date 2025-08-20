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
"""Test all Gaze functionality."""
import os
import re

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from pymovements import __version__
from pymovements import Events
from pymovements import Experiment
from pymovements import EyeTracker
from pymovements import Gaze
from pymovements import Screen
# PK


@pytest.mark.parametrize(
    'init_arg',
    [
        pytest.param(
            None,
            id='None',
        ),
        pytest.param(
            pl.DataFrame(),
            id='no_eye_velocity_columns',
        ),
    ],
)
def test_gaze_init(init_arg):
    gaze = Gaze(init_arg)
    assert isinstance(gaze.samples, pl.DataFrame)


@pytest.mark.parametrize(
    ('init_df', 'velocity_columns'),
    [
        pytest.param(
            pl.DataFrame(schema={'x_vel': pl.Float64, 'y_vel': pl.Float64}),
            ['x_vel', 'y_vel'],
            id='no_eye_velocity_columns',
        ),
        pytest.param(
            pl.DataFrame(schema={'abc': pl.Int64, 'x_vel': pl.Float64, 'y_vel': pl.Float64}),
            ['x_vel', 'y_vel'],
            id='no_eye_velocity_columns_with_other_columns',
        ),
        pytest.param(
            pl.DataFrame(schema={'x_right_vel': pl.Float64, 'y_right_vel': pl.Float64}),
            ['x_right_vel', 'y_right_vel'],
            id='right_eye_velocity_columns',
        ),
        pytest.param(
            pl.DataFrame(schema={'x_left_vel': pl.Float64, 'y_left_vel': pl.Float64}),
            ['x_left_vel', 'y_left_vel'],
            id='left_eye_velocity_columns',
        ),
        pytest.param(
            pl.DataFrame(
                schema={
                    'x_left_vel': pl.Float64, 'y_left_vel': pl.Float64,
                    'x_right_vel': pl.Float64, 'y_right_vel': pl.Float64,
                },
            ),
            ['x_left_vel', 'y_left_vel', 'x_right_vel', 'y_right_vel'],
            id='both_eyes_velocity_columns',
        ),
    ],
)
def test_gaze_velocity_columns(init_df, velocity_columns):
    gaze = Gaze(init_df, velocity_columns=velocity_columns)

    assert 'velocity' in gaze.columns


@pytest.mark.parametrize(
    ('init_df', 'pixel_columns'),
    [
        pytest.param(
            pl.DataFrame(schema={'x_pix': pl.Float64, 'y_pix': pl.Float64}),
            ['x_pix', 'y_pix'],
            id='no_eye_pix_pos_columns',
        ),
        pytest.param(
            pl.DataFrame(schema={'abc': pl.Int64, 'x_pix': pl.Float64, 'y_pix': pl.Float64}),
            ['x_pix', 'y_pix'],
            id='no_eye_pix_pos_columns_with_other_columns',
        ),
        pytest.param(
            pl.DataFrame(schema={'x_right_pix': pl.Float64, 'y_right_pix': pl.Float64}),
            ['x_right_pix', 'y_right_pix'],
            id='right_eye_pix_pos_columns',
        ),
        pytest.param(
            pl.DataFrame(schema={'x_left_pix': pl.Float64, 'y_left_pix': pl.Float64}),
            ['x_left_pix', 'y_left_pix'],
            id='left_eye_pix_pos_columns',
        ),
        pytest.param(
            pl.DataFrame(
                schema={
                    'x_left_pix': pl.Float64, 'y_left_pix': pl.Float64,
                    'x_right_pix': pl.Float64, 'y_right_pix': pl.Float64,
                },
            ),
            ['x_left_pix', 'y_left_pix', 'x_right_pix', 'y_right_pix'],
            id='both_eyes_pix_pos_columns',
        ),
    ],
)
def test_gaze_pixel_position_columns(init_df, pixel_columns):
    gaze = Gaze(init_df, pixel_columns=pixel_columns)

    assert 'pixel' in gaze.columns


@pytest.mark.parametrize(
    ('init_df', 'position_columns'),
    [
        pytest.param(
            pl.DataFrame(schema={'x_pos': pl.Float64, 'y_pos': pl.Float64}),
            ['x_pos', 'y_pos'],
            id='no_eye_pos_columns',
        ),
        pytest.param(
            pl.DataFrame(schema={'abc': pl.Int64, 'x_pos': pl.Float64, 'y_pos': pl.Float64}),
            ['x_pos', 'y_pos'],
            id='no_eye_pos_columns_with_other_columns',
        ),
        pytest.param(
            pl.DataFrame(schema={'x_right_pos': pl.Float64, 'y_right_pos': pl.Float64}),
            ['x_right_pos', 'y_right_pos'],
            id='right_eye_pos_columns',
        ),
        pytest.param(
            pl.DataFrame(schema={'x_left_pos': pl.Float64, 'y_left_pos': pl.Float64}),
            ['x_left_pos', 'y_left_pos'],
            id='left_eye_pos_columns',
        ),
        pytest.param(
            pl.DataFrame(
                schema={
                    'x_left_pos': pl.Float64, 'y_left_pos': pl.Float64,
                    'x_right_pos': pl.Float64, 'y_right_pos': pl.Float64,
                },
            ),
            ['x_left_pos', 'y_left_pos', 'x_right_pos', 'y_right_pos'],
            id='both_eyes_pos_columns',
        ),
    ],
)
def test_gaze_position_columns(init_df, position_columns):
    gaze = Gaze(init_df, position_columns=position_columns)

    assert 'position' in gaze.columns


def test_gaze_copy_with_experiment():
    gaze = Gaze(
        pl.DataFrame(schema={'x': pl.Float64, 'y': pl.Float64}),
        experiment=Experiment(1024, 768, 38, 30, 60, 'center', 1000),
        position_columns=['x', 'y'],
    )

    gaze_copy = gaze.clone()

    # We want to have separate dataframes but with the exact same data.
    assert gaze.samples is not gaze_copy.samples
    assert_frame_equal(gaze.samples, gaze_copy.samples)

    # We want to have separate experiment instances but the same values.
    assert gaze.experiment is not gaze_copy.experiment
    assert gaze.experiment == gaze_copy.experiment


def test_gaze_copy_no_experiment():
    gaze = Gaze(
        pl.DataFrame(schema={'x': pl.Float64, 'y': pl.Float64}),
        experiment=None,
        position_columns=['x', 'y'],
    )

    gaze_copy = gaze.clone()

    # We want to have separate dataframes but with the exact same data.
    assert gaze.samples is not gaze_copy.samples
    assert_frame_equal(gaze.samples, gaze_copy.samples)

    # We want to have separate experiment instances but the same values.
    assert gaze.experiment is gaze_copy.experiment


def test_gaze_is_copy():
    gaze = Gaze(
        pl.DataFrame(schema={'x': pl.Float64, 'y': pl.Float64}),
        experiment=None,
        position_columns=['x', 'y'],
    )

    gaze_copy = gaze.clone()

    assert gaze_copy is not gaze
    assert_frame_equal(gaze.samples, gaze_copy.samples)


def test_gaze_copy_events():
    gaze = Gaze(
        pl.DataFrame(schema={'x': pl.Float64, 'y': pl.Float64}),
        experiment=None,
        position_columns=['x', 'y'],
        events=Events(
            name='saccade',
            onsets=[0],
            offsets=[123],
        ),
    )

    gaze_copy = gaze.clone()

    assert gaze_copy.events is not gaze.events
    assert_frame_equal(gaze.events.frame, gaze_copy.events.frame)


def test_gaze_split():
    gaze = Gaze(
        pl.DataFrame(
            {
                'x': [0, 1, 2, 3],
                'y': [1, 1, 0, 0],
                'trial_id': [0, 1, 1, 2],
            },
            schema={'x': pl.Float64, 'y': pl.Float64, 'trial_id': pl.Int8},
        ),
        experiment=None,
        position_columns=['x', 'y'],
    )

    split_gaze = gaze.split('trial_id')
    assert all(gaze.samples.n_unique('trial_id') == 1 for gaze in split_gaze)
    assert len(split_gaze) == 3
    assert_frame_equal(gaze.samples.filter(pl.col('trial_id') == 0), split_gaze[0].samples)
    assert_frame_equal(gaze.samples.filter(pl.col('trial_id') == 1), split_gaze[1].samples)
    assert_frame_equal(gaze.samples.filter(pl.col('trial_id') == 2), split_gaze[2].samples)


def test_gaze_split_list():
    gaze = Gaze(
        pl.DataFrame(
            {
                'x': [0, 1, 2, 3],
                'y': [1, 1, 0, 0],
                'trial_ida': [0, 1, 1, 2],
                'trial_idb': ['a', 'b', 'c', 'c'],
            },
            schema={
                'x': pl.Float64,
                'y': pl.Float64,
                'trial_ida': pl.Int8,
                'trial_idb': pl.Utf8,
            },
        ),
        experiment=None,
        position_columns=['x', 'y'],
    )

    split_gaze = gaze.split(['trial_ida', 'trial_idb'])
    assert all(gaze.samples.n_unique(['trial_ida', 'trial_idb']) == 1 for gaze in split_gaze)
    assert len(split_gaze) == 4


def test_gaze_compute_event_properties_no_events():
    gaze = Gaze(
        pl.DataFrame(schema={'x': pl.Float64, 'y': pl.Float64, 'trial_id': pl.Int8}),
        position_columns=['x', 'y'],
        trial_columns=['trial_id'],
    )

    with pytest.warns(
        UserWarning,
        match='No events available to compute event properties. Did you forget to use detect()?',
    ):
        gaze.compute_event_properties('amplitude')


def test_gaze_dataframe_split_events():
    gaze = Gaze(
        pl.DataFrame(
            {
                'x': [0, 1, 2, 3],
                'y': [1, 1, 0, 0],
                'trial_id': [0, 1, 1, 2],
            },
            schema={'x': pl.Float64, 'y': pl.Float64, 'trial_id': pl.Int8},
        ),
        experiment=None,
        position_columns=['x', 'y'],
        events=Events(
            pl.DataFrame(
                {
                    'name': ['fixation', 'fixation', 'saccade', 'fixation'],
                    'onset': [0, 1, 2, 3],
                    'offset': [1, 2, 3, 4],
                    'trial_id': [0, 1, 1, 2],
                },
            ),
        ),
    )

    by = 'trial_id'
    split_gaze = gaze.split(by)
    assert all(gaze.events.frame.n_unique(by) == 1 for gaze in split_gaze)
    assert_frame_equal(gaze.events.frame.filter(pl.col(by) == 0), split_gaze[0].events.frame)
    assert_frame_equal(gaze.events.frame.filter(pl.col(by) == 1), split_gaze[1].events.frame)
    assert_frame_equal(gaze.events.frame.filter(pl.col(by) == 2), split_gaze[2].events.frame)


def test_gaze_dataframe_split_events_list():
    gaze = Gaze(
        pl.DataFrame(
            {
                'x': [0, 1, 2, 3],
                'y': [1, 1, 0, 0],
                'trial_ida': [0, 1, 1, 2],
                'trial_idb': [0, 1, 2, 2],
            },
        ),
        experiment=None,
        position_columns=['x', 'y'],
        events=Events(
            pl.DataFrame(
                {
                    'name': ['fixation', 'fixation', 'saccade', 'fixation'],
                    'onset': [0, 1, 2, 3],
                    'offset': [1, 2, 3, 4],
                    'trial_ida': [0, 1, 1, 2],
                    'trial_idb': [0, 1, 2, 2],
                },
            ),
        ),
    )

    by = ['trial_ida', 'trial_idb']
    split_gaze = gaze.split(by)
    assert len(split_gaze) == 4
    assert all(gaze.events.frame.n_unique(by) == 1 for gaze in split_gaze)


def test_gaze_dataframe_split_default():
    gaze = Gaze(
        pl.DataFrame(
            {
                'x': [0, 1, 2, 3],
                'y': [1, 1, 0, 0],
                'trial_id': [0, 1, 1, 2],
            },
            schema={'x': pl.Float64, 'y': pl.Float64, 'trial_id': pl.Int8},
        ),
        experiment=None,
        position_columns=['x', 'y'],
        events=Events(
            pl.DataFrame(
                {
                    'name': ['fixation', 'fixation', 'saccade', 'fixation'],
                    'onset': [0, 1, 2, 3],
                    'offset': [1, 2, 3, 4],
                    'trial_id': [0, 1, 1, 2],
                },
            ),
        ),
        trial_columns=['trial_id'],
    )

    by = 'trial_id'
    split_gaze = gaze.split()
    assert all(gaze.events.frame.n_unique(by) == 1 for gaze in split_gaze)
    assert_frame_equal(gaze.events.frame.filter(pl.col(by) == 0), split_gaze[0].events.frame)
    assert_frame_equal(gaze.events.frame.filter(pl.col(by) == 1), split_gaze[1].events.frame)
    assert_frame_equal(gaze.events.frame.filter(pl.col(by) == 2), split_gaze[2].events.frame)


def test_gaze_dataframe_split_default_no_trial_columns():
    gaze = Gaze(
        pl.DataFrame(
            {
                'x': [0, 1, 2, 3],
                'y': [1, 1, 0, 0],
                'trial_id': [0, 1, 1, 2],
            },
            schema={'x': pl.Float64, 'y': pl.Float64, 'trial_id': pl.Int8},
        ),
        experiment=None,
        position_columns=['x', 'y'],
        events=Events(
            pl.DataFrame(
                {
                    'name': ['fixation', 'fixation', 'saccade', 'fixation'],
                    'onset': [0, 1, 2, 3],
                    'offset': [1, 2, 3, 4],
                    'trial_id': [0, 1, 1, 2],
                },
            ),
        ),
    )

    with pytest.raises(TypeError):
        gaze.split()


@pytest.mark.parametrize(
    ('gaze', 'attribute'),
    [
        pytest.param(
            Gaze(),
            'frame',
            id='frame',
        ),
    ],
)
def test_dataset_definition_get_attribute_is_deprecated(gaze, attribute):
    with pytest.warns(DeprecationWarning):
        getattr(gaze, attribute)


@pytest.mark.parametrize(
    ('gaze', 'attribute', 'value'),
    [
        pytest.param(
            Gaze(),
            'frame',
            pl.DataFrame(),
            id='frame',
        ),
    ],
)
def test_gaze_set_attribute_is_deprecated(gaze, attribute, value):
    with pytest.warns(DeprecationWarning):
        setattr(gaze, attribute, value)


@pytest.mark.parametrize(
    'attribute',
    [
        'frame',
    ],
)
def test_gaze_get_attribute_is_removed(attribute):
    definition = Gaze()
    with pytest.raises(DeprecationWarning) as info:
        getattr(definition, attribute)

    regex = re.compile(r'.*will be removed in v(?P<version>[0-9]*[.][0-9]*[.][0-9]*)[.)].*')

    msg = info.value.args[0]
    remove_version = regex.match(msg).groupdict()['version']
    current_version = __version__.split('+')[0]
    assert current_version < remove_version, (
        f'Gaze.{attribute} was planned to be removed in v{remove_version}. '
        f'Current version is v{current_version}.'
    )


def _create_gaze():
    # Creating a Gaze object
    return Gaze(
        pl.DataFrame(
            {
                'x': [0, 1, 2, 3],
                'y': [1, 1, 0, 0],
                'trial_id': [0, 1, 1, 2],
            },
            schema={'x': pl.Float64, 'y': pl.Float64, 'trial_id': pl.Int8},
        ),
        experiment=Experiment(
            screen=Screen(
                width_px=1280, height_px=1024, width_cm=38.0, height_cm=30.0,
                distance_cm=68.0, origin='upper left',
            ), eyetracker=EyeTracker(
                sampling_rate=1000.0, left=None,
                right=None, model='MyModel', version=None, vendor=None, mount=None,
            ),
        ),
        position_columns=['x', 'y'],
        events=Events(
            pl.DataFrame(
                {
                    'name': ['fixation', 'fixation', 'saccade', 'fixation'],
                    'onset': [0, 1, 2, 3],
                    'offset': [1, 2, 3, 4],
                    'trial_id': [0, 1, 1, 2],
                },
            ),
        ),
    )


def test_gaze_save_csv(tmp_path):

    gaze = _create_gaze()
    # Saving Gaze to tmp_path
    gaze.save(
        dirname=tmp_path,
        verbose=2,
        extension='csv',
    )
    assert os.path.exists(tmp_path / 'samples.csv')
    assert os.path.exists(tmp_path / 'events.csv')
    assert os.path.exists(tmp_path / 'experiment.yaml')


def test_gaze_save_feather(tmp_path):
    gaze = _create_gaze()
    # Saving Gaze to tmp_path
    gaze.save(
        dirname=tmp_path,
        verbose=2,
        extension='feather',
    )
    assert os.path.exists(tmp_path / 'samples.feather')
    assert os.path.exists(tmp_path / 'events.feather')
    assert os.path.exists(tmp_path / 'experiment.yaml')


def test_gaze_save_no_events(tmp_path):

    gaze = _create_gaze()

    # Saving Gaze to tmp_path
    gaze.save(
        dirname=tmp_path,
        save_events=False,
        verbose=2,
        extension='csv',
    )
    assert not os.path.exists(tmp_path / 'events.csv')
    assert os.path.exists(tmp_path / 'samples.csv')
    assert os.path.exists(tmp_path / 'experiment.yaml')


def test_gaze_save_no_samples(tmp_path):

    gaze = _create_gaze()

    # Saving Gaze to tmp_path
    gaze.save(
        dirname=tmp_path,
        save_samples=False,
        verbose=2,
        extension='csv',
    )
    assert os.path.exists(tmp_path / 'events.csv')
    assert not os.path.exists(tmp_path / 'samples.csv')
    assert os.path.exists(tmp_path / 'experiment.yaml')


def test_gaze_save_no_experiment(tmp_path):

    gaze = _create_gaze()

    # Saving Gaze to tmp_path
    gaze.save(
        dirname=tmp_path,
        save_experiment=False,
        verbose=2,
        extension='csv',
    )
    assert os.path.exists(tmp_path / 'events.csv')
    assert os.path.exists(tmp_path / 'samples.csv')
    assert not os.path.exists(tmp_path / 'experiment.yaml')


def test_gaze_save_wrong_extension_events(tmp_path):
    gaze = _create_gaze()

    with pytest.raises(ValueError):
        gaze.save(
            dirname=tmp_path,
            verbose=1,
            extension='blabla',
        )

def test_gaze_save_wrong_extension_preprocessed(tmp_path):
    gaze = _create_gaze()

    with pytest.raises(ValueError):
        gaze.save(
            dirname=tmp_path,
            save_events=False,
            verbose=1,
            extension='blabla',
        )

def test_gaze_save_empty_experiment(tmp_path):
    gaze = _create_gaze()
    gaze.experiment = None

    gaze.save(
        dirname=tmp_path,
        verbose=1,
        extension='csv',
    )
    assert os.path.exists(tmp_path / 'events.csv')
    assert os.path.exists(tmp_path / 'samples.csv')
    assert not os.path.exists(tmp_path / 'experiment.yaml')
