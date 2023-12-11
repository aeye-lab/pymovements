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
"""Test GazeDataFrame detect method."""
import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

import pymovements as pm
from pymovements.synthetic import step_function


@pytest.mark.parametrize(
    ('method', 'kwargs', 'gaze', 'expected'),
    [
        pytest.param(
            'idt',
            {
                'dispersion_threshold': 1,
                'minimum_duration': 10,
            },
            pm.gaze.from_numpy(
                time=np.arange(0, 100, 1),
                time_unit='ms',
                position=np.stack([np.arange(0, 200, 2), np.arange(0, 200, 2)], axis=0),
                experiment=pm.Experiment(1024, 768, 38, 30, 60, 'center', 10),
            ),
            pm.events.EventDataFrame(),
            id='idt_constant_velocity_no_fixation',
        ),

        pytest.param(
            'idt',
            {
                'dispersion_threshold': 1,
                'minimum_duration': 2,
            },
            pm.gaze.from_numpy(
                position=step_function(length=100, steps=[0], values=[(0, 0)]),
                orient='row',
                experiment=pm.Experiment(1024, 768, 38, 30, 60, 'center', 10),
            ),
            pm.EventDataFrame(name='fixation', onsets=[0], offsets=[99]),
            id='idt_constant_position_single_fixation',
        ),

        pytest.param(
            'idt',
            {
                'dispersion_threshold': 1,
                'minimum_duration': 2,
                'name': 'custom_fixation',
            },
            pm.gaze.from_numpy(
                position=step_function(length=100, steps=[0], values=[(0, 0)]),
                orient='row',
                experiment=pm.Experiment(1024, 768, 38, 30, 60, 'center', 10),
            ),
            pm.EventDataFrame(name='custom_fixation', onsets=[0], offsets=[99]),
            id='idt_constant_position_single_fixation_custom_name',
        ),

        pytest.param(
            'idt',
            {
                'dispersion_threshold': 1,
                'minimum_duration': 2,
            },
            pm.gaze.from_numpy(
                position=step_function(
                    length=100, steps=[49, 50], values=[(9, 9), (1, 1)], start_value=(0, 0),
                ),
                orient='row',
                experiment=pm.Experiment(1024, 768, 38, 30, 60, 'center', 10),
            ),
            pm.events.EventDataFrame(name='fixation', onsets=[0, 50], offsets=[49, 99]),
            id='idt_three_steps_two_fixations',
        ),

        pytest.param(
            'idt',
            {
                'dispersion_threshold': 1,
                'minimum_duration': 2,
            },
            pm.gaze.from_numpy(
                position=step_function(
                    length=100, steps=[10, 20, 90],
                    values=[(np.nan, np.nan), (0, 0), (np.nan, np.nan)],
                ),
                orient='row',
                experiment=pm.Experiment(1024, 768, 38, 30, 60, 'center', 10),
            ),
            pm.events.EventDataFrame(name='fixation', onsets=[0, 20], offsets=[9, 89]),
            id='idt_two_fixations_interrupted_by_nan',
        ),

        pytest.param(
            'idt',
            {
                'dispersion_threshold': 1,
                'minimum_duration': 2,
                'include_nan': True,
            },
            pm.gaze.from_numpy(
                position=step_function(
                    length=100, steps=[10, 20, 90],
                    values=[(np.nan, np.nan), (0, 0), (np.nan, np.nan)],
                ),
                orient='row',
                experiment=pm.Experiment(1024, 768, 38, 30, 60, 'center', 10),
            ),
            pm.events.EventDataFrame(name='fixation', onsets=[0], offsets=[89]),
            id='idt_one_fixation_including_nan',
        ),

        pytest.param(
            'idt',
            {
                'dispersion_threshold': 1,
                'minimum_duration': 2,
            },
            pm.gaze.from_numpy(
                time=np.arange(1000, 1100, dtype=int),
                time_unit='ms',
                position=step_function(length=100, steps=[0], values=[(0, 0)]),
                orient='row',
                experiment=pm.Experiment(1024, 768, 38, 30, 60, 'center', 10),
            ),
            pm.events.EventDataFrame(
                name='fixation',
                onsets=[1000],
                offsets=[1099],
            ),
            id='idt_constant_position_single_fixation_with_timesteps_int',
        ),

        pytest.param(
            'idt',
            {
                'dispersion_threshold': 1,
                'minimum_duration': 2,
            },
            pm.gaze.from_numpy(
                time=np.arange(1000, 1010, 0.1, dtype=float),
                time_unit='ms',
                position=step_function(length=100, steps=[0], values=[(0, 0)]),
                orient='row',
                experiment=pm.Experiment(1024, 768, 38, 30, 60, 'center', 10),
            ),
            pm.events.EventDataFrame(
                name='fixation',
                onsets=[1000],
                offsets=[1099],
            ),
            id='idt_constant_position_single_fixation_with_timesteps_float',
            marks=pytest.mark.xfail(reason='#532'),
        ),

        pytest.param(
            'idt',
            {
                'dispersion_threshold': 1,
                'minimum_duration': 2,
            },
            pm.gaze.from_numpy(
                time=np.reshape(np.arange(1000, 1100, dtype=int), (100, 1)),
                time_unit='ms',
                position=step_function(length=100, steps=[0], values=[(0, 0)]),
                orient='row',
                experiment=pm.Experiment(1024, 768, 38, 30, 60, 'center', 10),
            ),
            pm.events.EventDataFrame(
                name='fixation',
                onsets=[1000],
                offsets=[1099],
            ),
            id='idt_constant_position_single_fixation_with_timesteps_int_extra_dim',
        ),

        pytest.param(
            'ivt',
            {
                'velocity_threshold': 1,
                'minimum_duration': 10,
            },
            pm.gaze.from_numpy(
                time=np.arange(0, 100, 1),
                time_unit='ms',
                velocity=np.ones((2, 100)) * 20,
                experiment=pm.Experiment(1024, 768, 38, 30, 60, 'center', 10),
            ),
            pm.events.EventDataFrame(),
            id='ivt_constant_velocity_no_fixation',
        ),

        pytest.param(
            'ivt',
            {
                'velocity_threshold': 1,
                'minimum_duration': 1,
            },
            pm.gaze.from_numpy(
                velocity=np.zeros((2, 100)),
                experiment=pm.Experiment(1024, 768, 38, 30, 60, 'center', 10),
            ),
            pm.EventDataFrame(name='fixation', onsets=[0], offsets=[99]),
            id='ivt_constant_position_single_fixation',
        ),

        pytest.param(
            'ivt',
            {
                'velocity_threshold': 1,
                'minimum_duration': 1,
                'name': 'custom_fixation',
            },
            pm.gaze.from_numpy(
                velocity=np.zeros((2, 100)),
                experiment=pm.Experiment(1024, 768, 38, 30, 60, 'center', 10),
            ),
            pm.EventDataFrame(name='custom_fixation', onsets=[0], offsets=[99]),
            id='ivt_constant_position_single_fixation_custom_name',
        ),

        pytest.param(
            'ivt',
            {
                'velocity_threshold': 1,
                'minimum_duration': 1,
            },
            pm.gaze.from_numpy(
                velocity=step_function(
                    length=100, steps=[49, 51], values=[(90, 90), (0, 0)], start_value=(0, 0),
                ),
                orient='row',
                experiment=pm.Experiment(1024, 768, 38, 30, 60, 'center', 10),
            ),
            pm.events.EventDataFrame(name='fixation', onsets=[0, 51], offsets=[48, 99]),
            id='ivt_three_steps_two_fixations',
        ),

        pytest.param(
            'ivt',
            {
                'velocity_threshold': 1,
                'minimum_duration': 1,
            },
            pm.gaze.from_numpy(
                velocity=step_function(
                    length=100, steps=[10, 20, 90],
                    values=[(np.nan, np.nan), (0, 0), (np.nan, np.nan)],
                ),
                orient='row',
                experiment=pm.Experiment(1024, 768, 38, 30, 60, 'center', 10),
            ),
            pm.events.EventDataFrame(name='fixation', onsets=[0, 20], offsets=[9, 89]),
            id='ivt_two_fixations_interrupted_by_nan',
        ),

        pytest.param(
            'ivt',
            {
                'velocity_threshold': 1,
                'minimum_duration': 1,
                'include_nan': True,
            },
            pm.gaze.from_numpy(
                velocity=step_function(
                    length=100, steps=[10, 20, 90],
                    values=[(np.nan, np.nan), (0, 0), (np.nan, np.nan)],
                ),
                orient='row',
                experiment=pm.Experiment(1024, 768, 38, 30, 60, 'center', 10),
            ),
            pm.events.EventDataFrame(name='fixation', onsets=[0], offsets=[89]),
            id='ivt_one_fixation_including_nan',
        ),

        pytest.param(
            'ivt',
            {
                'velocity_threshold': 1,
                'minimum_duration': 2,
            },
            pm.gaze.from_numpy(
                time=np.arange(1000, 1100, dtype=int),
                time_unit='ms',
                velocity=step_function(length=100, steps=[0], values=[(0, 0)]),
                orient='row',
                experiment=pm.Experiment(1024, 768, 38, 30, 60, 'center', 10),
            ),
            pm.events.EventDataFrame(
                name='fixation',
                onsets=[1000],
                offsets=[1099],
            ),
            id='ivt_constant_position_single_fixation_with_timesteps_int',
        ),

        pytest.param(
            'ivt',
            {
                'velocity_threshold': 1,
                'minimum_duration': 2,
            },
            pm.gaze.from_numpy(
                time=np.arange(1000, 1010, 0.1, dtype=float),
                time_unit='ms',
                velocity=step_function(length=100, steps=[0], values=[(0, 0)]),
                orient='row',
                experiment=pm.Experiment(1024, 768, 38, 30, 60, 'center', 10),
            ),
            pm.events.EventDataFrame(
                name='fixation',
                onsets=[1000],
                offsets=[1099],
            ),
            id='ivt_constant_position_single_fixation_with_timesteps_float',
            marks=pytest.mark.xfail(reason='#532'),
        ),

        pytest.param(
            'ivt',
            {
                'velocity_threshold': 1,
                'minimum_duration': 2,
            },
            pm.gaze.from_numpy(
                time=np.reshape(np.arange(1000, 1100, dtype=int), (100, 1)),
                time_unit='ms',
                velocity=step_function(length=100, steps=[0], values=[(0, 0)]),
                orient='row',
                experiment=pm.Experiment(1024, 768, 38, 30, 60, 'center', 10),
            ),
            pm.events.EventDataFrame(
                name='fixation',
                onsets=[1000],
                offsets=[1099],
            ),
            id='ivt_constant_position_single_fixation_with_timesteps_int_extra_dim',
        ),

        pytest.param(
            'ivt',
            {
                'velocity_threshold': 1,
                'minimum_duration': 2,
                'eye': 'auto',
            },
            pm.gaze.from_numpy(
                velocity=step_function(length=100, steps=[0], values=[(0, 0)]),
                orient='row',
                experiment=pm.Experiment(1024, 768, 38, 30, 60, 'center', 10),
            ),
            pm.events.EventDataFrame(
                name='fixation',
                onsets=[0],
                offsets=[99],
            ),
            id='ivt_constant_position_binocular_fixation_two_components_eye_auto',
        ),

        pytest.param(
            'ivt',
            {
                'velocity_threshold': 1,
                'minimum_duration': 2,
                'eye': 'auto',
            },
            pm.gaze.from_numpy(
                velocity=step_function(length=100, steps=[0], values=[(0, 0, 0, 0)]),
                orient='row',
                experiment=pm.Experiment(1024, 768, 38, 30, 60, 'center', 10),
            ),
            pm.events.EventDataFrame(
                name='fixation',
                onsets=[0],
                offsets=[99],
            ),
            id='ivt_constant_position_binocular_fixation_four_components_eye_auto',
        ),

        pytest.param(
            'ivt',
            {
                'velocity_threshold': 1,
                'minimum_duration': 2,
                'eye': 'auto',
            },
            pm.gaze.from_numpy(
                velocity=step_function(length=100, steps=[0], values=[(0, 0, 0, 0, 0, 0)]),
                orient='row',
                experiment=pm.Experiment(1024, 768, 38, 30, 60, 'center', 10),
            ),
            pm.events.EventDataFrame(
                name='fixation',
                onsets=[0],
                offsets=[99],
            ),
            id='ivt_constant_position_binocular_fixation_six_components_eye_auto',
        ),

        pytest.param(
            'ivt',
            {
                'velocity_threshold': 1,
                'minimum_duration': 2,
                'eye': 'left',
            },
            pm.gaze.from_numpy(
                velocity=step_function(
                    length=100, steps=[0, 10], values=[(0, 0, 1, 1, 1, 1), (0, 0, 0, 0, 0, 0)],
                ),
                orient='row',
                experiment=pm.Experiment(1024, 768, 38, 30, 60, 'center', 10),
            ),
            pm.events.EventDataFrame(
                name='fixation',
                onsets=[0],
                offsets=[99],
            ),
            id='ivt_constant_position_monocular_fixation_six_components_eye_left',
        ),

        pytest.param(
            'ivt',
            {
                'velocity_threshold': 1,
                'minimum_duration': 2,
                'eye': 'right',
            },
            pm.gaze.from_numpy(
                velocity=step_function(
                    length=100, steps=[0, 10], values=[(1, 1, 0, 0, 1, 1), (0, 0, 0, 0, 0, 0)],
                ),
                orient='row',
                experiment=pm.Experiment(1024, 768, 38, 30, 60, 'center', 10),
            ),
            pm.events.EventDataFrame(
                name='fixation',
                onsets=[0],
                offsets=[99],
            ),
            id='ivt_constant_position_monocular_fixation_six_components_eye_right',
        ),

        pytest.param(
            'ivt',
            {
                'velocity_threshold': 1,
                'minimum_duration': 2,
                'eye': 'cyclops',
            },
            pm.gaze.from_numpy(
                velocity=step_function(
                    length=100, steps=[0, 10], values=[(1, 1, 1, 1, 0, 0), (0, 0, 0, 0, 0, 0)],
                ),
                orient='row',
                experiment=pm.Experiment(1024, 768, 38, 30, 60, 'center', 10),
            ),
            pm.events.EventDataFrame(
                name='fixation',
                onsets=[0],
                offsets=[99],
            ),
            id='ivt_constant_position_monocular_fixation_six_components_eye_cyclops',
        ),

        pytest.param(
            'microsaccades',
            {
                'threshold': 10,
            },
            pm.gaze.from_numpy(
                time=np.reshape(np.arange(1000, 1100, dtype=int), (100, 1)),
                time_unit='ms',
                velocity=step_function(length=100, steps=[40, 50], values=[(9, 9), (0, 0)]),
                orient='row',
                experiment=pm.Experiment(1024, 768, 38, 30, 60, 'center', 10),
            ),
            pm.EventDataFrame(),
            id='microsaccades_two_steps_one_saccade_high_threshold_no_events',
        ),

        pytest.param(
            'microsaccades',
            {
                'threshold': 1e-5,
            },
            pm.gaze.from_numpy(
                velocity=step_function(length=100, steps=[40, 50], values=[(9, 9), (0, 0)]),
                orient='row',
                experiment=pm.Experiment(1024, 768, 38, 30, 60, 'center', 10),
            ),
            pm.EventDataFrame(
                name='saccade',
                onsets=[40],
                offsets=[49],
            ),
            id='microsaccades_two_steps_one_saccade',
        ),

        pytest.param(
            'microsaccades',
            {
                'threshold': 1e-5,
                'name': 'custom_saccade',
            },
            pm.gaze.from_numpy(
                velocity=step_function(length=100, steps=[40, 50], values=[(9, 9), (0, 0)]),
                orient='row',
                experiment=pm.Experiment(1024, 768, 38, 30, 60, 'center', 10),
            ),
            pm.EventDataFrame(
                name='custom_saccade',
                onsets=[40],
                offsets=[49],
            ),
            id='microsaccades_two_steps_one_saccade_custom_name',
        ),

        pytest.param(
            'microsaccades',
            {
                'threshold': 1e-5,
            },
            pm.gaze.from_numpy(
                velocity=step_function(
                    length=100,
                    steps=[20, 30, 70, 80],
                    values=[(9, 9), (0, 0), (9, 9), (0, 0)],
                    start_value=(0, 0),
                ),
                orient='row',
                experiment=pm.Experiment(1024, 768, 38, 30, 60, 'center', 10),
            ),
            pm.EventDataFrame(
                name='saccade',
                onsets=[20, 70],
                offsets=[29, 79],
            ),
            id='microsaccades_four_steps_two_saccades',
        ),

        pytest.param(
            'microsaccades',
            {
                'threshold': 1,
                'include_nan': True,
            },
            pm.gaze.from_numpy(
                velocity=step_function(
                    length=100,
                    steps=[20, 25, 28, 30, 70, 80],
                    values=[(9, 9), (np.nan, np.nan), (9, 9), (0, 0), (9, 9), (0, 0)],
                    start_value=(0, 0),
                ),
                orient='row',
                experiment=pm.Experiment(1024, 768, 38, 30, 60, 'center', 10),
            ),
            pm.EventDataFrame(
                name='saccade',
                onsets=[20, 70],
                offsets=[29, 79],
            ),
            id='microsaccades_four_steps_two_saccades_nan_delete_ending_leading_nan',
        ),

        pytest.param(
            'microsaccades',
            {
                'threshold': 1,
                'minimum_duration': 1,
            },
            pm.gaze.from_numpy(
                velocity=step_function(
                    length=100,
                    steps=[20, 25, 28, 30, 70, 80],
                    values=[(9, 9), (np.nan, np.nan), (9, 9), (0, 0), (9, 9), (0, 0)],
                    start_value=(0, 0),
                ),
                orient='row',
                experiment=pm.Experiment(1024, 768, 38, 30, 60, 'center', 10),
            ),
            pm.EventDataFrame(
                name='saccade',
                onsets=[20, 28, 70],
                offsets=[24, 29, 79],
            ),
            id='microsaccades_three_saccades_nan_delete_ending_leading_nan',
        ),

        pytest.param(
            'microsaccades',
            {
                'threshold': 1e-5,
                'minimum_duration': 1,
            },
            pm.gaze.from_numpy(
                time=np.arange(1000, 1100, dtype=int),
                time_unit='ms',
                velocity=step_function(
                    length=100,
                    steps=[40, 50],
                    values=[(9, 9), (0, 0)],
                    start_value=(0, 0),
                ),
                orient='row',
                experiment=pm.Experiment(1024, 768, 38, 30, 60, 'center', 10),
            ),
            pm.EventDataFrame(
                name='saccade',
                onsets=[1040],
                offsets=[1049],
            ),
            id='microsaccades_two_steps_one_saccade_timesteps',
        ),

        pytest.param(
            'microsaccades',
            {
                'threshold': 'std',
            },
            pm.gaze.from_numpy(
                time=np.arange(1000, 1100, dtype=int),
                time_unit='ms',
                velocity=step_function(
                    length=100,
                    steps=[40, 50],
                    values=[(9, 9), (0, 0)],
                    start_value=(0, 0),
                ),
                orient='row',
                experiment=pm.Experiment(1024, 768, 38, 30, 60, 'center', 10),
            ),
            pm.EventDataFrame(),
            id='microsaccades_two_steps_one_saccade_timesteps',
        ),

        pytest.param(
            'fill',
            {},
            pm.gaze.from_numpy(
                time=np.arange(0, 100),
                time_unit='ms',
                events=pm.EventDataFrame(name='fixation', onsets=[0], offsets=[100]),
            ),
            pm.EventDataFrame(name='fixation', onsets=[0], offsets=[100]),
            id='fill_fixation_from_start_to_end_no_fill',
        ),

        pytest.param(
            'fill',
            {},
            pm.gaze.from_numpy(
                time=np.arange(0, 100),
                time_unit='ms',
                events=pm.EventDataFrame(name='fixation', onsets=[10], offsets=[100]),
            ),
            pm.EventDataFrame(
                name=['fixation', 'unclassified'],
                onsets=[10, 0],
                offsets=[100, 9],
            ),
            id='fill_fixation_10_ms_after_start_to_end_single_fill',
        ),

        pytest.param(
            'fill',
            {},
            pm.gaze.from_numpy(
                time=np.arange(0, 100),
                time_unit='ms',
                events=pm.EventDataFrame(name='fixation', onsets=[0], offsets=[90]),
            ),
            pm.EventDataFrame(
                name=['fixation', 'unclassified'],
                onsets=[0, 90],
                offsets=[90, 99],
            ),
            id='fill_fixation_from_start_to_10_ms_before_end_single_fill',
        ),

        pytest.param(
            'fill',
            {},
            pm.gaze.from_numpy(
                time=np.arange(0, 100),
                time_unit='ms',
                events=pm.EventDataFrame(name='fixation', onsets=[0, 50], offsets=[40, 100]),
            ),
            pm.EventDataFrame(
                name=['fixation', 'fixation', 'unclassified'],
                onsets=[0, 50, 40],
                offsets=[40, 100, 49],
            ),
            id='fill_fixation_10_ms_break_at_40ms_single_fill',
        ),

        pytest.param(
            'fill',
            {},
            pm.gaze.from_numpy(
                time=np.arange(0, 100),
                time_unit='ms',
                events=pm.EventDataFrame(
                    name=['fixation', 'saccade'], onsets=[0, 50], offsets=[40, 100],
                ),
            ),
            pm.EventDataFrame(
                name=['fixation', 'saccade', 'unclassified'],
                onsets=[0, 50, 40],
                offsets=[40, 100, 49],
            ),
            id='fill_fixation_10_ms_break_then_saccade_until_end_single_fill',
        ),
    ],
)
def test_gaze_detect(method, kwargs, gaze, expected):
    gaze.detect(method, **kwargs)
    assert_frame_equal(gaze.events.frame, expected.frame)


@pytest.mark.parametrize(
    ('method', 'kwargs', 'gaze', 'exception', 'exception_msg'),
    [
        pytest.param(
            'ivt',
            {
                'velocity_threshold': 1,
                'minimum_duration': 2,
                'eye': 'auto',
            },
            pm.gaze.GazeDataFrame(None, pm.Experiment(1024, 768, 38, 30, 60, 'center', 10)),
            pl.exceptions.ColumnNotFoundError,
            "Column 'velocity' not found. Available columns are: []",
            id='ivt_no_velocity_raises_column_not_found_error',
        ),

        pytest.param(
            'ivt',
            {
                'velocity_threshold': 1,
                'minimum_duration': 2,
                'eye': 'left',
            },
            pm.gaze.from_numpy(
                velocity=step_function(length=100, steps=[0], values=[(0, 0)]),
                orient='row',
                experiment=pm.Experiment(1024, 768, 38, 30, 60, 'center', 10),
            ),
            AttributeError,
            'left eye is only supported for data with at least 4 components',
            id='ivt_left_eye_two_components_raises_attribute_error',
        ),

        pytest.param(
            'ivt',
            {
                'velocity_threshold': 1,
                'minimum_duration': 2,
                'eye': 'right',
            },
            pm.gaze.from_numpy(
                velocity=step_function(length=100, steps=[0], values=[(0, 0)]),
                orient='row',
                experiment=pm.Experiment(1024, 768, 38, 30, 60, 'center', 10),
            ),
            AttributeError,
            'right eye is only supported for data with at least 4 components',
            id='ivt_right_eye_two_components_raises_attribute_error',
        ),

        pytest.param(
            'ivt',
            {
                'velocity_threshold': 1,
                'minimum_duration': 2,
                'eye': 'cyclops',
            },
            pm.gaze.from_numpy(
                velocity=step_function(length=100, steps=[0], values=[(0, 0, 0, 0)]),
                orient='row',
                experiment=pm.Experiment(1024, 768, 38, 30, 60, 'center', 10),
            ),
            AttributeError,
            'cyclops eye is only supported for data with at least 6 components',
            id='ivt_cyclops_eye_four_components_raises_attribute_error',
        ),

        pytest.param(
            'ivt',
            {
                'velocity_threshold': 1,
                'minimum_duration': 2,
                'eye': 'foobar',
            },
            pm.gaze.from_numpy(
                velocity=step_function(length=100, steps=[0], values=[(0, 0, 0, 0)]),
                orient='row',
                experiment=pm.Experiment(1024, 768, 38, 30, 60, 'center', 10),
            ),
            ValueError,
            "unknown eye 'foobar'. Supported values are: ['auto', 'left', 'right', 'cyclops']",
            id='ivt_cyclops_eye_four_components_raises_attribute_error',
        ),
    ],
)
def test_gaze_detect_raises_exception(method, kwargs, gaze, exception, exception_msg):
    with pytest.raises(exception) as exc_info:
        gaze.detect(method, **kwargs)

    msg, = exc_info.value.args
    assert msg == exception_msg
