from typing import List, Optional, Type

import pandas as pd
import numpy as np
import numbers


def verify_split_ratio(ratio: float) -> float:
    """Verifies if ratio for split is acceptable (float in range(0, 1)).

    Args:
        ratio (float): ratio in which data is to be split

    Returns:
        ratio if value is acceptable, else raises ValueError
    """
    if not isinstance(ratio, float):
        raise TypeError("Expected ratio to be of type float")

    if ratio <= 0 or ratio >= 1:
        raise ValueError("Split ratio should be in between 0 and 1")
    
    return ratio


def verify_stratification(
    data: pd.DataFrame,
    ratio: float,
    user_col: str,
    item_col: str,
    filter_col: str,
    min_rating_count: Optional[int] = 1,
    seed: Optional[int] = 11
):
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Expected data to be of type pandas.DataFrame")
    
    verify_split_ratio(ratio)

    if user_col not in data.columns:
        raise ValueError(f"Column: {user_col} not in data")
    
    if item_col not in data.columns:
        raise ValueError(f"Column: {item_col} not in data")

    if filter_col != "user" and filter_col != "item":
        raise ValueError("Expected filter_col to be either user or item")

    if min_rating_count < 1:
        raise ValueError("min_rating_count should be greater than or equal to 1")


def ratio_split(data: pd.DataFrame, ratios: List[float], shuffle: Optional[bool]=False, seed: Optional[int]=42):
    indices = np.cumsum(ratios).tolist()[: -1]

    if shuffle:
        data = data.sample(frac=1, random_state=seed)

    splits = np.split(data, [round(x * len(data)) for x in indices])

    for i in range(len(ratios)):
        splits[i]["split_index"] = i
    
    return splits


def estimate_batches(input_size, batch_size):
    return int(np.ceil(input_size / batch_size))


def get_rng(seed):
    if seed is None:
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('{} can not be used to create a numpy.random.RandomState'.format(seed))