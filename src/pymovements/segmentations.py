from decimal import Decimal

import numpy as np


def segmentation_by_bins(
        arr: np.ndarray,
        n_bins: int,
        v_min: float,
        v_max: float,
) -> np.ndarray:
    v_min = Decimal(v_min)
    v_max = Decimal(v_max)

    sequence_length = arr.shape[0]
    segmentation = np.zeros((n_bins, sequence_length), dtype=bool)

    bin_width = (v_max - v_min) / Decimal(n_bins)

    bins = [
        (v_min + Decimal(bin_idx) * bin_width, v_min + Decimal(bin_idx + 1) * bin_width)
        for bin_idx in range(n_bins)
    ]

    for bin_idx, (bin_min, bin_max) in enumerate(bins):
        segmentation[bin_idx] = np.logical_and(
            arr >= bin_min, arr < bin_max,
        )
    segmentation[-1] = np.logical_or(
        segmentation[-1], arr == v_max,
    )

    return segmentation
