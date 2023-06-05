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
"""Test GazeDataFrame initialization."""
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from pymovements.gaze.gaze_dataframe import GazeDataFrame


@pytest.mark.parametrize(
    ('init_kwargs', 'expected_frame'),
    [
        pytest.param(
            {
                'data': pl.DataFrame(),
            },
            pl.DataFrame(),
            id='empty_df_no_schema',
        ),

        pytest.param(
            {
                'data': pl.DataFrame(schema={'abc': pl.Int64}),
            },
            pl.DataFrame(schema={'abc': pl.Int64}),
            id='empty_df_with_schema_no_component_columns',
        ),

        pytest.param(
            {
                'data': pl.DataFrame(schema={'x': pl.Float64, 'y': pl.Float64}),
                'pixel_columns': ['x', 'y'],
            },
            pl.DataFrame(schema={'pixel': pl.List(pl.Float64)}),
            id='empty_df_with_schema_two_pixel_columns',
        ),

        pytest.param(
            {
                'data': pl.DataFrame(schema={'abc': pl.Int64, 'x': pl.Float64, 'y': pl.Float64}),
                'pixel_columns': ['x', 'y'],
            },
            pl.DataFrame(schema={'abc': pl.Int64, 'pixel': pl.List(pl.Float64)}),
            id='empty_df_with_three_column_schema_two_pixel_columns',
        ),

        pytest.param(
            {
                'data': pl.DataFrame(
                    schema={
                        'xr': pl.Float64, 'yr': pl.Float64, 'xl': pl.Float64, 'yl': pl.Float64,
                    },
                ),
                'pixel_columns': ['xr', 'yr', 'xl', 'yl'],
            },
            pl.DataFrame(schema={'pixel': pl.List(pl.Float64)}),
            id='empty_df_with_schema_four_pixel_columns',
        ),

        pytest.param(
            {
                'data': pl.DataFrame(
                    schema={
                        'x_right': pl.Float64, 'y_right': pl.Float64,
                        'x_left': pl.Float64, 'y_left': pl.Float64,
                        'x_avg': pl.Float64, 'y_avg': pl.Float64,
                    },
                ),
                'pixel_columns': [
                    'x_right', 'y_right', 'x_left', 'y_left', 'x_avg', 'y_avg',
                ],
            },
            pl.DataFrame(schema={'pixel': pl.List(pl.Float64)}),
            id='empty_df_with_schema_six_pixel_columns',
        ),

        pytest.param(
            {
                'data': pl.from_dict(
                    {'x': [1.23], 'y': [4.56]}, schema={'x': pl.Float64, 'y': pl.Float64},
                ),
                'pixel_columns': ['x', 'y'],
            },
            pl.from_dict(
                {'pixel': [[1.23, 4.56]]},
                schema={'pixel': pl.List(pl.Float64)},
            ),
            id='df_single_row_two_pixel_columns',
        ),

        pytest.param(
            {
                'data': pl.from_dict(
                    {'abc': [1], 'x': [1.23], 'y': [4.56]},
                    schema={'abc': pl.Int64, 'x': pl.Float64, 'y': pl.Float64},
                ),
                'pixel_columns': ['x', 'y'],
            },
            pl.from_dict(
                {'abc': [1], 'pixel': [[1.23, 4.56]]},
                schema={'abc': pl.Int64, 'pixel': pl.List(pl.Float64)},
            ),
            id='df_single_row_three_columns_two_pixel_columns',
        ),

        pytest.param(
            {
                'data': pl.from_dict(
                    {'xl': [1.2], 'yl': [3.4], 'xr': [5.6], 'yr': [7.8]},
                    schema={'xl': pl.Float64, 'yl': pl.Float64, 'xr': pl.Float64, 'yr': pl.Float64},
                ),
                'pixel_columns': ['xl', 'yl', 'xr', 'yr'],
            },
            pl.from_dict(
                {'pixel': [[1.2, 3.4, 5.6, 7.8]]},
                schema={'pixel': pl.List(pl.Float64)},
            ),
            id='df_single_row_four_pixel_columns',
        ),

        pytest.param(
            {
                'data': pl.from_dict(
                    {
                        'x_right': [0.1], 'y_right': [0.2],
                        'x_left': [0.3], 'y_left': [0.4],
                        'x_avg': [0.5], 'y_avg': [0.6],
                    },
                    schema={
                        'x_right': pl.Float64, 'y_right': pl.Float64,
                        'x_left': pl.Float64, 'y_left': pl.Float64,
                        'x_avg': pl.Float64, 'y_avg': pl.Float64,
                    },
                ),
                'pixel_columns': [
                    'x_right', 'y_right', 'x_left', 'y_left', 'x_avg', 'y_avg',
                ],
            },
            pl.from_dict(
                {'pixel': [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]]},
                schema={'pixel': pl.List(pl.Float64)},
            ),
            id='df_single_row_six_pixel_columns',
        ),

        pytest.param(
            {
                'data': pl.DataFrame(schema={'x': pl.Float64, 'y': pl.Float64}),
                'position_columns': ['x', 'y'],
            },
            pl.DataFrame(schema={'position': pl.List(pl.Float64)}),
            id='empty_df_with_schema_two_position_columns',
        ),

        pytest.param(
            {
                'data': pl.DataFrame(schema={'abc': pl.Int64, 'x': pl.Float64, 'y': pl.Float64}),
                'position_columns': ['x', 'y'],
            },
            pl.DataFrame(schema={'abc': pl.Int64, 'position': pl.List(pl.Float64)}),
            id='empty_df_with_three_column_schema_two_position_columns',
        ),

        pytest.param(
            {
                'data': pl.DataFrame(
                    schema={
                        'xr': pl.Float64, 'yr': pl.Float64, 'xl': pl.Float64, 'yl': pl.Float64,
                    },
                ),
                'position_columns': ['xr', 'yr', 'xl', 'yl'],
            },
            pl.DataFrame(schema={'position': pl.List(pl.Float64)}),
            id='empty_df_with_schema_four_position_columns',
        ),

        pytest.param(
            {
                'data': pl.DataFrame(
                    schema={
                        'x_right': pl.Float64, 'y_right': pl.Float64,
                        'x_left': pl.Float64, 'y_left': pl.Float64,
                        'x_avg': pl.Float64, 'y_avg': pl.Float64,
                    },
                ),
                'position_columns': [
                    'x_right', 'y_right', 'x_left', 'y_left', 'x_avg', 'y_avg',
                ],
            },
            pl.DataFrame(schema={'position': pl.List(pl.Float64)}),
            id='empty_df_with_schema_six_position_columns',
        ),

        pytest.param(
            {
                'data': pl.from_dict(
                    {'x': [1.23], 'y': [4.56]}, schema={'x': pl.Float64, 'y': pl.Float64},
                ),
                'position_columns': ['x', 'y'],
            },
            pl.from_dict(
                {'position': [[1.23, 4.56]]},
                schema={'position': pl.List(pl.Float64)},
            ),
            id='df_single_row_two_position_columns',
        ),

        pytest.param(
            {
                'data': pl.from_dict(
                    {'abc': [1], 'x': [1.23], 'y': [4.56]},
                    schema={'abc': pl.Int64, 'x': pl.Float64, 'y': pl.Float64},
                ),
                'position_columns': ['x', 'y'],
            },
            pl.from_dict(
                {'abc': [1], 'position': [[1.23, 4.56]]},
                schema={'abc': pl.Int64, 'position': pl.List(pl.Float64)},
            ),
            id='df_single_row_three_columns_two_position_columns',
        ),

        pytest.param(
            {
                'data': pl.from_dict(
                    {'xl': [1.2], 'yl': [3.4], 'xr': [5.6], 'yr': [7.8]},
                    schema={'xl': pl.Float64, 'yl': pl.Float64, 'xr': pl.Float64, 'yr': pl.Float64},
                ),
                'position_columns': ['xl', 'yl', 'xr', 'yr'],
            },
            pl.from_dict(
                {'position': [[1.2, 3.4, 5.6, 7.8]]},
                schema={'position': pl.List(pl.Float64)},
            ),
            id='df_single_row_four_position_columns',
        ),

        pytest.param(
            {
                'data': pl.from_dict(
                    {
                        'x_right': [0.1], 'y_right': [0.2],
                        'x_left': [0.3], 'y_left': [0.4],
                        'x_avg': [0.5], 'y_avg': [0.6],
                    },
                    schema={
                        'x_right': pl.Float64, 'y_right': pl.Float64,
                        'x_left': pl.Float64, 'y_left': pl.Float64,
                        'x_avg': pl.Float64, 'y_avg': pl.Float64,
                    },
                ),
                'position_columns': [
                    'x_right', 'y_right', 'x_left', 'y_left', 'x_avg', 'y_avg',
                ],
            },
            pl.from_dict(
                {'position': [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]]},
                schema={'position': pl.List(pl.Float64)},
            ),
            id='df_single_row_six_position_columns',
        ),

        pytest.param(
            {
                'data': pl.DataFrame(schema={'x_vel': pl.Float64, 'y_vel': pl.Float64}),
                'velocity_columns': ['x_vel', 'y_vel'],
            },
            pl.DataFrame(schema={'velocity': pl.List(pl.Float64)}),
            id='empty_df_with_schema_two_velocity_columns',
        ),

        pytest.param(
            {
                'data': pl.DataFrame(
                    schema={
                        'abc': pl.Int64, 'x_vel': pl.Float64, 'y_vel': pl.Float64,
                    },
                ),
                'velocity_columns': ['x_vel', 'y_vel'],
            },
            pl.DataFrame(schema={'abc': pl.Int64, 'velocity': pl.List(pl.Float64)}),
            id='empty_df_with_three_column_schema_two_velocity_columns',
        ),

        pytest.param(
            {
                'data': pl.DataFrame(
                    schema={
                        'xr_vel': pl.Float64, 'yr_vel': pl.Float64,
                        'xl_vel': pl.Float64, 'yl_vel': pl.Float64,
                    },
                ),
                'velocity_columns': ['xr_vel', 'yr_vel', 'xl_vel', 'yl_vel'],
            },
            pl.DataFrame(schema={'velocity': pl.List(pl.Float64)}),
            id='empty_df_with_schema_four_velocity_columns',
        ),

        pytest.param(
            {
                'data': pl.DataFrame(
                    schema={
                        'x_right_vel': pl.Float64, 'y_right_vel': pl.Float64,
                        'x_left_vel': pl.Float64, 'y_left_vel': pl.Float64,
                        'x_avg_vel': pl.Float64, 'y_avg_vel': pl.Float64,
                    },
                ),
                'velocity_columns': [
                    'x_right_vel', 'y_right_vel',
                    'x_left_vel', 'y_left_vel',
                    'x_avg_vel', 'y_avg_vel',
                ],
            },
            pl.DataFrame(schema={'velocity': pl.List(pl.Float64)}),
            id='empty_df_with_schema_six_velocity_columns',
        ),

        pytest.param(
            {
                'data': pl.from_dict(
                    {'x_vel': [1.23], 'y_vel': [4.56]},
                    schema={'x_vel': pl.Float64, 'y_vel': pl.Float64},
                ),
                'velocity_columns': ['x_vel', 'y_vel'],
            },
            pl.from_dict(
                {'velocity': [[1.23, 4.56]]},
                schema={'velocity': pl.List(pl.Float64)},
            ),
            id='df_single_row_two_velocity_columns',
        ),

        pytest.param(
            {
                'data': pl.from_dict(
                    {'abc': [1], 'x_vel': [1.23], 'y_vel': [4.56]},
                    schema={'abc': pl.Int64, 'x_vel': pl.Float64, 'y_vel': pl.Float64},
                ),
                'velocity_columns': ['x_vel', 'y_vel'],
            },
            pl.from_dict(
                {'abc': [1], 'velocity': [[1.23, 4.56]]},
                schema={'abc': pl.Int64, 'velocity': pl.List(pl.Float64)},
            ),
            id='df_single_row_three_columns_two_velocity_columns',
        ),

        pytest.param(
            {
                'data': pl.from_dict(
                    {
                        'xl_vel': [1.2], 'yl_vel': [3.4],
                        'xr_vel': [5.6], 'yr_vel': [7.8],
                    },
                    schema={
                        'xl_vel': pl.Float64, 'yl_vel': pl.Float64,
                        'xr_vel': pl.Float64, 'yr_vel': pl.Float64,
                    },
                ),
                'velocity_columns': ['xl_vel', 'yl_vel', 'xr_vel', 'yr_vel'],
            },
            pl.from_dict(
                {'velocity': [[1.2, 3.4, 5.6, 7.8]]},
                schema={'velocity': pl.List(pl.Float64)},
            ),
            id='df_single_row_four_velocity_columns',
        ),

        pytest.param(
            {
                'data': pl.from_dict(
                    {
                        'x_right_vel': [0.1], 'y_right_vel': [0.2],
                        'x_left_vel': [0.3], 'y_left_vel': [0.4],
                        'x_avg_vel': [0.5], 'y_avg_vel': [0.6],
                    },
                    schema={
                        'x_right_vel': pl.Float64, 'y_right_vel': pl.Float64,
                        'x_left_vel': pl.Float64, 'y_left_vel': pl.Float64,
                        'x_avg_vel': pl.Float64, 'y_avg_vel': pl.Float64,
                    },
                ),
                'velocity_columns': [
                    'x_right_vel', 'y_right_vel',
                    'x_left_vel', 'y_left_vel',
                    'x_avg_vel', 'y_avg_vel',
                ],
            },
            pl.from_dict(
                {'velocity': [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]]},
                schema={'velocity': pl.List(pl.Float64)},
            ),
            id='df_single_row_six_velocity_columns',
        ),

        pytest.param(
            {
                'data': pl.DataFrame(schema={'x_acc': pl.Float64, 'y_acc': pl.Float64}),
                'acceleration_columns': ['x_acc', 'y_acc'],
            },
            pl.DataFrame(schema={'acceleration': pl.List(pl.Float64)}),
            id='empty_df_with_schema_two_acceleration_columns',
        ),

        pytest.param(
            {
                'data': pl.DataFrame(
                    schema={
                        'abc': pl.Int64, 'x_acc': pl.Float64, 'y_acc': pl.Float64,
                    },
                ),
                'acceleration_columns': ['x_acc', 'y_acc'],
            },
            pl.DataFrame(schema={'abc': pl.Int64, 'acceleration': pl.List(pl.Float64)}),
            id='empty_df_with_three_column_schema_two_acceleration_columns',
        ),

        pytest.param(
            {
                'data': pl.DataFrame(
                    schema={
                        'xr_acc': pl.Float64, 'yr_acc': pl.Float64,
                        'xl_acc': pl.Float64, 'yl_acc': pl.Float64,
                    },
                ),
                'acceleration_columns': ['xr_acc', 'yr_acc', 'xl_acc', 'yl_acc'],
            },
            pl.DataFrame(schema={'acceleration': pl.List(pl.Float64)}),
            id='empty_df_with_schema_four_acceleration_columns',
        ),

        pytest.param(
            {
                'data': pl.DataFrame(
                    schema={
                        'x_right_acc': pl.Float64, 'y_right_acc': pl.Float64,
                        'x_left_acc': pl.Float64, 'y_left_acc': pl.Float64,
                        'x_avg_acc': pl.Float64, 'y_avg_acc': pl.Float64,
                    },
                ),
                'acceleration_columns': [
                    'x_right_acc', 'y_right_acc',
                    'x_left_acc', 'y_left_acc',
                    'x_avg_acc', 'y_avg_acc',
                ],
            },
            pl.DataFrame(schema={'acceleration': pl.List(pl.Float64)}),
            id='empty_df_with_schema_six_acceleration_columns',
        ),

        pytest.param(
            {
                'data': pl.from_dict(
                    {'x_acc': [1.23], 'y_acc': [4.56]},
                    schema={'x_acc': pl.Float64, 'y_acc': pl.Float64},
                ),
                'acceleration_columns': ['x_acc', 'y_acc'],
            },
            pl.from_dict(
                {'acceleration': [[1.23, 4.56]]},
                schema={'acceleration': pl.List(pl.Float64)},
            ),
            id='df_single_row_two_acceleration_columns',
        ),

        pytest.param(
            {
                'data': pl.from_dict(
                    {'abc': [1], 'x_acc': [1.23], 'y_acc': [4.56]},
                    schema={'abc': pl.Int64, 'x_acc': pl.Float64, 'y_acc': pl.Float64},
                ),
                'acceleration_columns': ['x_acc', 'y_acc'],
            },
            pl.from_dict(
                {'abc': [1], 'acceleration': [[1.23, 4.56]]},
                schema={'abc': pl.Int64, 'acceleration': pl.List(pl.Float64)},
            ),
            id='df_single_row_three_columns_two_acceleration_columns',
        ),

        pytest.param(
            {
                'data': pl.from_dict(
                    {
                        'xl_acc': [1.2], 'yl_acc': [3.4],
                        'xr_acc': [5.6], 'yr_acc': [7.8],
                    },
                    schema={
                        'xl_acc': pl.Float64, 'yl_acc': pl.Float64,
                        'xr_acc': pl.Float64, 'yr_acc': pl.Float64,
                    },
                ),
                'acceleration_columns': ['xl_acc', 'yl_acc', 'xr_acc', 'yr_acc'],
            },
            pl.from_dict(
                {'acceleration': [[1.2, 3.4, 5.6, 7.8]]},
                schema={'acceleration': pl.List(pl.Float64)},
            ),
            id='df_single_row_four_acceleration_columns',
        ),

        pytest.param(
            {
                'data': pl.from_dict(
                    {
                        'x_right_acc': [0.1], 'y_right_acc': [0.2],
                        'x_left_acc': [0.3], 'y_left_acc': [0.4],
                        'x_avg_acc': [0.5], 'y_avg_acc': [0.6],
                    },
                    schema={
                        'x_right_acc': pl.Float64, 'y_right_acc': pl.Float64,
                        'x_left_acc': pl.Float64, 'y_left_acc': pl.Float64,
                        'x_avg_acc': pl.Float64, 'y_avg_acc': pl.Float64,
                    },
                ),
                'acceleration_columns': [
                    'x_right_acc', 'y_right_acc',
                    'x_left_acc', 'y_left_acc',
                    'x_avg_acc', 'y_avg_acc',
                ],
            },
            pl.from_dict(
                {'acceleration': [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]]},
                schema={'acceleration': pl.List(pl.Float64)},
            ),
            id='df_single_row_six_acceleration_columns',
        ),

        pytest.param(
            {
                'data': pl.from_dict(
                    {
                        'x_pix': [0.1], 'y_pix': [0.2],
                        'x_dva': [1.1], 'y_dva': [1.2],
                        'x_vel': [3.1], 'y_vel': [3.2],
                        'x_acc': [5.1], 'y_acc': [5.2],
                    },
                    schema={
                        'x_pix': pl.Float64, 'y_pix': pl.Float64,
                        'x_dva': pl.Float64, 'y_dva': pl.Float64,
                        'x_vel': pl.Float64, 'y_vel': pl.Float64,
                        'x_acc': pl.Float64, 'y_acc': pl.Float64,
                    },
                ),
                'pixel_columns': ['x_pix', 'y_pix'],
                'position_columns': ['x_dva', 'y_dva'],
                'velocity_columns': ['x_vel', 'y_vel'],
                'acceleration_columns': ['x_acc', 'y_acc'],
            },
            pl.from_dict(
                {
                    'pixel': [[0.1, 0.2]],
                    'position': [[1.1, 1.2]],
                    'velocity': [[3.1, 3.2]],
                    'acceleration': [[5.1, 5.2]],
                },
                schema={
                    'pixel': pl.List(pl.Float64),
                    'position': pl.List(pl.Float64),
                    'velocity': pl.List(pl.Float64),
                    'acceleration': pl.List(pl.Float64),
                },
            ),
            id='df_single_row_all_types_two_columns',
        ),

        pytest.param(
            {
                'data': pl.from_dict(
                    {
                        'x_right_pos_pix': [0.1], 'y_right_pos_pix': [0.2],
                        'x_left_pos_pix': [0.3], 'y_left_pos_pix': [0.4],
                        'x_avg_pos_pix': [0.5], 'y_avg_pos_pix': [0.6],
                        'x_right_pos_dva': [1.1], 'y_right_pos_dva': [1.2],
                        'x_left_pos_dva': [1.3], 'y_left_pos_dva': [1.4],
                        'x_avg_pos_dva': [1.5], 'y_avg_pos_dva': [1.6],
                        'x_right_vel_dva': [3.1], 'y_right_vel_dva': [3.2],
                        'x_left_vel_dva': [3.3], 'y_left_vel_dva': [3.4],
                        'x_avg_vel_dva': [3.5], 'y_avg_vel_dva': [3.6],
                        'x_right_acc_dva': [5.1], 'y_right_acc_dva': [5.2],
                        'x_left_acc_dva': [5.3], 'y_left_acc_dva': [5.4],
                        'x_avg_acc_dva': [5.5], 'y_avg_acc_dva': [5.6],
                    },
                    schema={
                        'x_right_pos_pix': pl.Float64, 'y_right_pos_pix': pl.Float64,
                        'x_left_pos_pix': pl.Float64, 'y_left_pos_pix': pl.Float64,
                        'x_avg_pos_pix': pl.Float64, 'y_avg_pos_pix': pl.Float64,
                        'x_right_pos_dva': pl.Float64, 'y_right_pos_dva': pl.Float64,
                        'x_left_pos_dva': pl.Float64, 'y_left_pos_dva': pl.Float64,
                        'x_avg_pos_dva': pl.Float64, 'y_avg_pos_dva': pl.Float64,
                        'x_right_vel_dva': pl.Float64, 'y_right_vel_dva': pl.Float64,
                        'x_left_vel_dva': pl.Float64, 'y_left_vel_dva': pl.Float64,
                        'x_avg_vel_dva': pl.Float64, 'y_avg_vel_dva': pl.Float64,
                        'x_right_acc_dva': pl.Float64, 'y_right_acc_dva': pl.Float64,
                        'x_left_acc_dva': pl.Float64, 'y_left_acc_dva': pl.Float64,
                        'x_avg_acc_dva': pl.Float64, 'y_avg_acc_dva': pl.Float64,
                    },
                ),
                'pixel_columns': [
                    'x_right_pos_pix', 'y_right_pos_pix',
                    'x_left_pos_pix', 'y_left_pos_pix',
                    'x_avg_pos_pix', 'y_avg_pos_pix',
                ],
                'position_columns': [
                    'x_right_pos_dva', 'y_right_pos_dva',
                    'x_left_pos_dva', 'y_left_pos_dva',
                    'x_avg_pos_dva', 'y_avg_pos_dva',
                ],
                'velocity_columns': [
                    'x_right_vel_dva', 'y_right_vel_dva',
                    'x_left_vel_dva', 'y_left_vel_dva',
                    'x_avg_vel_dva', 'y_avg_vel_dva',
                ],
                'acceleration_columns': [
                    'x_right_acc_dva', 'y_right_acc_dva',
                    'x_left_acc_dva', 'y_left_acc_dva',
                    'x_avg_acc_dva', 'y_avg_acc_dva',
                ],
            },
            pl.from_dict(
                {
                    'pixel': [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]],
                    'position': [[1.1, 1.2, 1.3, 1.4, 1.5, 1.6]],
                    'velocity': [[3.1, 3.2, 3.3, 3.4, 3.5, 3.6]],
                    'acceleration': [[5.1, 5.2, 5.3, 5.4, 5.5, 5.6]],
                },
                schema={
                    'pixel': pl.List(pl.Float64),
                    'position': pl.List(pl.Float64),
                    'velocity': pl.List(pl.Float64),
                    'acceleration': pl.List(pl.Float64),
                },
            ),
            id='df_single_row_all_types_six_columns',
        ),
    ],
)
def test_init_gaze_dataframe_has_expected_frame(init_kwargs, expected_frame):
    gaze = GazeDataFrame(**init_kwargs)
    assert_frame_equal(gaze.frame, expected_frame)


@pytest.mark.parametrize(
    ('init_kwargs', 'exception', 'exception_msg'),
    [
        pytest.param(
            {
                'data': pl.DataFrame(schema={'x': pl.Float64, 'y': pl.Float64}),
                'pixel_columns': 1,
            },
            TypeError,
            'pixel_columns must be of type list, but is of type int',
            id='pixel_columns_int',
        ),

        pytest.param(
            {
                'data': pl.DataFrame(schema={'x': pl.Float64, 'y': pl.Float64}),
                'pixel_columns': 'x',
            },
            TypeError,
            'pixel_columns must be of type list, but is of type str',
            id='pixel_columns_str',
        ),

        pytest.param(
            {
                'data': pl.DataFrame(schema={'x': pl.Float64, 'y': pl.Float64}),
                'pixel_columns': [],
            },
            ValueError,
            'pixel_columns must contain either 2, 4 or 6 columns, but has 0',
            id='pixel_columns_empty_list',
        ),

        pytest.param(
            {
                'data': pl.DataFrame(schema={'x': pl.Float64, 'y': pl.Float64}),
                'pixel_columns': [0, 1],
            },
            TypeError,
            'all elements in pixel_columns must be of type str,'
            ' but one of the elements is of type int',
            id='pixel_columns_list_elements_not_string',
        ),

        pytest.param(
            {
                'data': pl.DataFrame(schema={'x': pl.Float64, 'y': pl.Float64}),
                'pixel_columns': ['x'],
            },
            ValueError,
            'pixel_columns must contain either 2, 4 or 6 columns, but has 1',
            id='pixel_columns_list_of_one',
        ),

        pytest.param(
            {
                'data': pl.DataFrame(
                    schema={
                        'xr': pl.Float64, 'yr': pl.Float64, 'xl': pl.Float64, 'yl': pl.Float64,
                    },
                ),
                'pixel_columns': ['xr', 'xl', 'yl'],
            },
            ValueError,
            'pixel_columns must contain either 2, 4 or 6 columns, but has 3',
            id='pixel_columns_list_of_three',
        ),

        pytest.param(
            {
                'data': pl.DataFrame(
                    schema={
                        'xr': pl.Float64, 'yr': pl.Float64,
                        'xl': pl.Float64, 'yl': pl.Float64,
                        'xa': pl.Float64, 'ya': pl.Float64,
                    },
                ),
                'pixel_columns': ['xr', 'yr', 'xl', 'yl', 'xa'],
            },
            ValueError,
            'pixel_columns must contain either 2, 4 or 6 columns, but has 5',
            id='pixel_columns_list_of_five',
        ),

        pytest.param(
            {
                'data': pl.DataFrame(
                    schema={
                        'abc': pl.Int64,
                        'xr': pl.Float64, 'yr': pl.Float64,
                        'xl': pl.Float64, 'yl': pl.Float64,
                        'xa': pl.Float64, 'ya': pl.Float64,
                    },
                ),
                'pixel_columns': ['xr', 'yr', 'xl', 'yl', 'xa', 'ya', 'abc'],
            },
            ValueError,
            'pixel_columns must contain either 2, 4 or 6 columns, but has 7',
            id='pixel_columns_list_of_seven',
        ),

        pytest.param(
            {
                'data': pl.DataFrame(schema={'x': pl.Float64, 'y': pl.Int64}),
                'pixel_columns': ['x', 'y'],
            },
            ValueError,
            'all columns in pixel_columns must be of same type, but types are'
            " ['Float64', 'Int64']",
            id='pixel_columns_different_type',
        ),

        pytest.param(
            {
                'data': pl.DataFrame(schema={'x': pl.Float64}),
                'pixel_columns': ['x', 'y'],
            },
            pl.exceptions.ColumnNotFoundError,
            'column y from pixel_columns is not available in dataframe',
            id='pixel_columns_missing_column',
        ),

        pytest.param(
            {
                'data': pl.DataFrame(schema={'x': pl.Float64, 'y': pl.Float64}),
                'position_columns': 1,
            },
            TypeError,
            'position_columns must be of type list, but is of type int',
            id='position_columns_int',
        ),

        pytest.param(
            {
                'data': pl.DataFrame(schema={'x': pl.Float64, 'y': pl.Float64}),
                'position_columns': 'x',
            },
            TypeError,
            'position_columns must be of type list, but is of type str',
            id='position_columns_str',
        ),

        pytest.param(
            {
                'data': pl.DataFrame(schema={'x': pl.Float64, 'y': pl.Float64}),
                'position_columns': [],
            },
            ValueError,
            'position_columns must contain either 2, 4 or 6 columns, but has 0',
            id='position_columns_empty_list',
        ),

        pytest.param(
            {
                'data': pl.DataFrame(schema={'x': pl.Float64, 'y': pl.Float64}),
                'position_columns': [0, 1],
            },
            TypeError,
            'all elements in position_columns must be of type str,'
            ' but one of the elements is of type int',
            id='position_columns_list_elements_not_string',
        ),

        pytest.param(
            {
                'data': pl.DataFrame(schema={'x': pl.Float64, 'y': pl.Float64}),
                'position_columns': ['x'],
            },
            ValueError,
            'position_columns must contain either 2, 4 or 6 columns, but has 1',
            id='position_columns_list_of_one',
        ),

        pytest.param(
            {
                'data': pl.DataFrame(
                    schema={
                        'xr': pl.Float64, 'yr': pl.Float64, 'xl': pl.Float64, 'yl': pl.Float64,
                    },
                ),
                'position_columns': ['xr', 'xl', 'yl'],
            },
            ValueError,
            'position_columns must contain either 2, 4 or 6 columns, but has 3',
            id='position_columns_list_of_three',
        ),

        pytest.param(
            {
                'data': pl.DataFrame(
                    schema={
                        'xr': pl.Float64, 'yr': pl.Float64,
                        'xl': pl.Float64, 'yl': pl.Float64,
                        'xa': pl.Float64, 'ya': pl.Float64,
                    },
                ),
                'position_columns': ['xr', 'yr', 'xl', 'yl', 'xa'],
            },
            ValueError,
            'position_columns must contain either 2, 4 or 6 columns, but has 5',
            id='position_columns_list_of_five',
        ),

        pytest.param(
            {
                'data': pl.DataFrame(
                    schema={
                        'abc': pl.Int64,
                        'xr': pl.Float64, 'yr': pl.Float64,
                        'xl': pl.Float64, 'yl': pl.Float64,
                        'xa': pl.Float64, 'ya': pl.Float64,
                    },
                ),
                'position_columns': ['xr', 'yr', 'xl', 'yl', 'xa', 'ya', 'abc'],
            },
            ValueError,
            'position_columns must contain either 2, 4 or 6 columns, but has 7',
            id='position_columns_list_of_seven',
        ),

        pytest.param(
            {
                'data': pl.DataFrame(schema={'x': pl.Float64, 'y': pl.Int64}),
                'position_columns': ['x', 'y'],
            },
            ValueError,
            'all columns in position_columns must be of same type, but types are'
            " ['Float64', 'Int64']",
            id='position_columns_different_type',
        ),

        pytest.param(
            {
                'data': pl.DataFrame(schema={'x': pl.Float64}),
                'position_columns': ['x', 'y'],
            },
            pl.exceptions.ColumnNotFoundError,
            'column y from position_columns is not available in dataframe',
            id='position_columns_missing_column',
        ),

        pytest.param(
            {
                'data': pl.DataFrame(schema={'x_vel': pl.Float64, 'y_vel': pl.Float64}),
                'velocity_columns': 1,
            },
            TypeError,
            'velocity_columns must be of type list, but is of type int',
            id='velocity_columns_int',
        ),

        pytest.param(
            {
                'data': pl.DataFrame(schema={'x_vel': pl.Float64, 'y_vel': pl.Float64}),
                'velocity_columns': 'x_vel',
            },
            TypeError,
            'velocity_columns must be of type list, but is of type str',
            id='velocity_columns_str',
        ),

        pytest.param(
            {
                'data': pl.DataFrame(schema={'x_vel': pl.Float64, 'y_vel': pl.Float64}),
                'velocity_columns': [],
            },
            ValueError,
            'velocity_columns must contain either 2, 4 or 6 columns, but has 0',
            id='velocity_columns_empty_list',
        ),

        pytest.param(
            {
                'data': pl.DataFrame(schema={'x_vel': pl.Float64, 'y_vel': pl.Float64}),
                'velocity_columns': [0, 1],
            },
            TypeError,
            'all elements in velocity_columns must be of type str,'
            ' but one of the elements is of type int',
            id='velocity_columns_list_elements_not_string',
        ),

        pytest.param(
            {
                'data': pl.DataFrame(schema={'x_vel': pl.Float64, 'y_vel': pl.Float64}),
                'velocity_columns': ['x_vel'],
            },
            ValueError,
            'velocity_columns must contain either 2, 4 or 6 columns, but has 1',
            id='velocity_columns_list_of_one',
        ),

        pytest.param(
            {
                'data': pl.DataFrame(
                    schema={
                        'xr_vel': pl.Float64, 'yr_vel': pl.Float64,
                        'xl_vel': pl.Float64, 'yl_vel': pl.Float64,
                    },
                ),
                'velocity_columns': ['xr_vel', 'xl_vel', 'yl_vel'],
            },
            ValueError,
            'velocity_columns must contain either 2, 4 or 6 columns, but has 3',
            id='velocity_columns_list_of_three',
        ),

        pytest.param(
            {
                'data': pl.DataFrame(
                    schema={
                        'xr_vel': pl.Float64, 'yr_vel': pl.Float64,
                        'xl_vel': pl.Float64, 'yl_vel': pl.Float64,
                        'xa_vel': pl.Float64, 'ya_vel': pl.Float64,
                    },
                ),
                'velocity_columns': ['xr_vel', 'yr_vel', 'xl_vel', 'yl_vel', 'xa_vel'],
            },
            ValueError,
            'velocity_columns must contain either 2, 4 or 6 columns, but has 5',
            id='velocity_columns_list_of_five',
        ),

        pytest.param(
            {
                'data': pl.DataFrame(
                    schema={
                        'abc': pl.Int64,
                        'xr_vel': pl.Float64, 'yr_vel': pl.Float64,
                        'xl_vel': pl.Float64, 'yl_vel': pl.Float64,
                        'xa_vel': pl.Float64, 'ya_vel': pl.Float64,
                    },
                ),
                'velocity_columns': [
                    'xr_vel', 'yr_vel', 'xl_vel', 'yl_vel', 'xa_vel', 'ya_vel', 'abc',
                ],
            },
            ValueError,
            'velocity_columns must contain either 2, 4 or 6 columns, but has 7',
            id='velocity_columns_list_of_seven',
        ),

        pytest.param(
            {
                'data': pl.DataFrame(schema={'x_vel': pl.Float64, 'y_vel': pl.Int64}),
                'velocity_columns': ['x_vel', 'y_vel'],
            },
            ValueError,
            'all columns in velocity_columns must be of same type, but types are'
            " ['Float64', 'Int64']",
            id='velocity_columns_different_type',
        ),

        pytest.param(
            {
                'data': pl.DataFrame(schema={'x_vel': pl.Float64}),
                'velocity_columns': ['x_vel', 'y_vel'],
            },
            pl.exceptions.ColumnNotFoundError,
            'column y_vel from velocity_columns is not available in dataframe',
            id='velocity_columns_missing_column',
        ),

        pytest.param(
            {
                'data': pl.DataFrame(schema={'x_acc': pl.Float64, 'y_acc': pl.Float64}),
                'acceleration_columns': 1,
            },
            TypeError,
            'acceleration_columns must be of type list, but is of type int',
            id='acceleration_columns_int',
        ),

        pytest.param(
            {
                'data': pl.DataFrame(schema={'x_acc': pl.Float64, 'y_acc': pl.Float64}),
                'acceleration_columns': 'x_acc',
            },
            TypeError,
            'acceleration_columns must be of type list, but is of type str',
            id='acceleration_columns_str',
        ),

        pytest.param(
            {
                'data': pl.DataFrame(schema={'x_acc': pl.Float64, 'y_acc': pl.Float64}),
                'acceleration_columns': [],
            },
            ValueError,
            'acceleration_columns must contain either 2, 4 or 6 columns, but has 0',
            id='acceleration_columns_empty_list',
        ),

        pytest.param(
            {
                'data': pl.DataFrame(schema={'x_acc': pl.Float64, 'y_acc': pl.Float64}),
                'acceleration_columns': [0, 1],
            },
            TypeError,
            'all elements in acceleration_columns must be of type str,'
            ' but one of the elements is of type int',
            id='acceleration_columns_list_elements_not_string',
        ),

        pytest.param(
            {
                'data': pl.DataFrame(schema={'x_acc': pl.Float64, 'y_acc': pl.Float64}),
                'acceleration_columns': ['x_acc'],
            },
            ValueError,
            'acceleration_columns must contain either 2, 4 or 6 columns, but has 1',
            id='acceleration_columns_list_of_one',
        ),

        pytest.param(
            {
                'data': pl.DataFrame(
                    schema={
                        'xr_acc': pl.Float64, 'yr_acc': pl.Float64,
                        'xl_acc': pl.Float64, 'yl_acc': pl.Float64,
                    },
                ),
                'acceleration_columns': ['xr_acc', 'xl_acc', 'yl_acc'],
            },
            ValueError,
            'acceleration_columns must contain either 2, 4 or 6 columns, but has 3',
            id='acceleration_columns_list_of_three',
        ),

        pytest.param(
            {
                'data': pl.DataFrame(
                    schema={
                        'xr_acc': pl.Float64, 'yr_acc': pl.Float64,
                        'xl_acc': pl.Float64, 'yl_acc': pl.Float64,
                        'xa_acc': pl.Float64, 'ya_acc': pl.Float64,
                    },
                ),
                'acceleration_columns': ['xr_acc', 'yr_acc', 'xl_acc', 'yl_acc', 'xa_acc'],
            },
            ValueError,
            'acceleration_columns must contain either 2, 4 or 6 columns, but has 5',
            id='acceleration_columns_list_of_five',
        ),

        pytest.param(
            {
                'data': pl.DataFrame(
                    schema={
                        'abc': pl.Int64,
                        'xr_acc': pl.Float64, 'yr_acc': pl.Float64,
                        'xl_acc': pl.Float64, 'yl_acc': pl.Float64,
                        'xa_acc': pl.Float64, 'ya_acc': pl.Float64,
                    },
                ),
                'acceleration_columns': [
                    'xr_acc', 'yr_acc', 'xl_acc', 'yl_acc', 'xa_acc', 'ya_acc', 'abc',
                ],
            },
            ValueError,
            'acceleration_columns must contain either 2, 4 or 6 columns, but has 7',
            id='acceleration_columns_list_of_seven',
        ),

        pytest.param(
            {
                'data': pl.DataFrame(schema={'x_acc': pl.Float64, 'y_acc': pl.Int64}),
                'acceleration_columns': ['x_acc', 'y_acc'],
            },
            ValueError,
            'all columns in acceleration_columns must be of same type, but types are'
            " ['Float64', 'Int64']",
            id='acceleration_columns_different_type',
        ),

        pytest.param(
            {
                'data': pl.DataFrame(schema={'x_acc': pl.Float64}),
                'acceleration_columns': ['x_acc', 'y_acc'],
            },
            pl.exceptions.ColumnNotFoundError,
            'column y_acc from acceleration_columns is not available in dataframe',
            id='acceleration_columns_missing_column',
        ),
    ],
)
def test_event_dataframe_init_exceptions(init_kwargs, exception, exception_msg):
    with pytest.raises(exception) as excinfo:
        GazeDataFrame(**init_kwargs)

    msg, = excinfo.value.args
    assert msg == exception_msg
