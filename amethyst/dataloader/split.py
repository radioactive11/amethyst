import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from data_utils import verify_split_ratio

def random_split(data: pd.DataFrame, split_ratio: float):
    ratio = verify_split_ratio(split_ratio)

    return train_test_split(data, test_size=None, train_size=ratio)



