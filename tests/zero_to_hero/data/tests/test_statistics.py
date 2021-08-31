"""
Testing the statistics applied during the pre-processing of the data
"""
from typing import Optional

import numpy as np
import pytest

from zero_to_hero.data.statistics import standardize_data


@pytest.mark.parametrize(
    "data, mean, std, return_moments",
    [
        (
            np.random.random((10, 4)),
            None,
            None,
            True,
        ),
        (
            np.random.random((2, 5)),
            None,
            None,
            True,
        ),
        (
            np.array([[0, 1.5], [1, 2]]),
            np.array([[0.5, 1.75]]),
            np.array([[0.5, 0.25]]),
            False,
        ),
    ],
)
def test_standardize_data(
    data: np.ndarray,
    mean: Optional[np.ndarray],
    std: Optional[np.ndarray],
    return_moments: bool,
) -> None:
    """
    Testing the standardize_data function
    :param data: np.ndarray
    :param mean: np.ndarray
    :param std: np.ndarray
    :param return_moments: boolean
    :return: None
    """
    if return_moments:
        standardized_data, mean, std = standardize_data(
            data=data,
            mean=mean,
            std=std,
            return_moments=return_moments,
        )
    else:
        standardized_data = standardize_data(
            data=data,
            mean=mean,
            std=std,
            return_moments=return_moments,
        )

    target_mean = np.mean(data, axis=0)
    target_std = np.sqrt(np.mean((data - data.mean(axis=0)) ** 2, axis=0))

    assert np.isclose(mean, target_mean).all()
    assert np.isclose(std, target_std).all()
    assert np.isclose(standardized_data, (data - target_mean) / target_std).all()
