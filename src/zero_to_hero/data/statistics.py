"""
Statistics needed for pre-processing the data
"""
from typing import Optional, Tuple, Union

import numpy as np


def standardize_data(
    data: np.ndarray,
    mean: Optional[np.ndarray] = None,
    std: Optional[np.ndarray] = None,
    return_moments: bool = False,
) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
    """
    Standarize data
    :param data: np.ndarray, data to standardize
    :param mean: np.ndarray, mean of data
    :param std: np.ndarray, standard deviation of data
    :param return_moments: boolean, (default True), to return the statistical moments or not
    :return: standardized data, (mean, std if return_moments == True)
    """
    if mean is None or std is None:
        mean = data.mean(0)
        std = data.std(0)
    data = (data - mean) / std
    if return_moments:
        return data, mean, std
    return data


def standardize_step(
    train_data: np.ndarray,
    valid_data: np.ndarray,
    test_data: np.ndarray,
    predict_data: Optional[np.ndarray] = None,
) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """

    :param train_data: np.ndarray, Training data
    :param valid_data: np.ndarray, Validation data
    :param test_data: np.ndarray, Test data
    :param predict_data: np.ndarray, Predict data
    :return:
        Train data: np.ndarray
        Valid data: np.ndarray
        Test_data: np.ndarray
        Predict_data if predict_data is not None: np.ndarray
    """
    train_data, mean, std = standardize_data(data=train_data, return_moments=True)

    data = standardize_data(data=valid_data, mean=mean, std=std)
    assert isinstance(data, np.ndarray)
    valid_data = data.copy()

    data = standardize_data(data=test_data, mean=mean, std=std)
    assert isinstance(data, np.ndarray)
    test_data = data.copy()

    if predict_data is None:
        return train_data, valid_data, test_data

    assert isinstance(predict_data, np.ndarray)
    data = standardize_data(data=predict_data, mean=mean, std=std)
    assert isinstance(data, np.ndarray)
    predict_data = data.copy()

    return train_data, valid_data, test_data, predict_data
