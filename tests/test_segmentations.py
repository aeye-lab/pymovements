from typing import Any, Dict

import numpy as np
import pytest

from pymovements.segmentations import segmentation_by_bins


# TODO: create fixture out of this
linspace_arr_10_bins_expected = np.zeros((10, 101), dtype=bool)
for bin_idx in range(10):
    bin_start = 10 * bin_idx
    bin_end = 10 * (bin_idx + 1)
    linspace_arr_10_bins_expected[bin_idx, bin_start:bin_end] = True
linspace_arr_10_bins_expected[0, 0] = True
linspace_arr_10_bins_expected[-1, -1] = True
linspace_arr_10_bins_expected[2, 30] = True
linspace_arr_10_bins_expected[3, 30] = False
linspace_arr_10_bins_expected[5, 60] = True
linspace_arr_10_bins_expected[6, 60] = False


@pytest.mark.parametrize(
    'params, expected',
    [
        pytest.param(
            {'arr': np.zeros(1000), 'n_bins': 2, 'v_min': 0, 'v_max': 1},
            np.stack([np.ones(1000, dtype=bool), np.zeros(1000, dtype=bool)]),
            id='all_zero_arr_2_bins_vmin=0',
        ),
        pytest.param(
            {'arr': np.zeros(1000), 'n_bins': 3, 'v_min': -1, 'v_max': 1},
            np.stack([np.zeros(1000, dtype=bool), np.ones(1000, dtype=bool),
                      np.zeros(1000, dtype=bool)]),
            id='all_zero_arr_3_bins_vmin=-1',
        ),
        pytest.param(
            {'arr': np.ones(1000), 'n_bins': 3, 'v_min': -1, 'v_max': 1},
            np.stack([np.zeros(1000, dtype=bool), np.zeros(1000, dtype=bool),
                      np.ones(1000, dtype=bool)]),
            id='all_one_arr_3_bins_vmin=-1',
        ),
        pytest.param(
            {'arr': np.linspace(0, 1, 101), 'n_bins': 10, 'v_min': 0, 'v_max': 1},
            linspace_arr_10_bins_expected,
            id='linspace_arr_10_bins_vmin=0',
        ),
    ]
)
def test_segmentation_by_bins(params: Dict[str, Any], expected: Any):
    segmentation = segmentation_by_bins(**params)

    segmentation_sums = np.sum(segmentation, axis=0)
    assert np.all(segmentation_sums == 1), np.where(np.all(np.sum(segmentation, axis=0) != 1))

    assert np.all(segmentation == expected), np.where(segmentation != expected)
