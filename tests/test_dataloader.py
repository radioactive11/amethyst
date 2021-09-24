import pandas as pd
import numpy as np
import pytest
import os

from amethyst.dataloader import split

BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, "samples")

class TestDataloader():
    def test_ratio(self):
        out_of_bound_ratio = 1.1
        wrong_dtype_ratio = 'a'
        correct_ratio = 0.75
        data = pd.read_csv(f"{DATA_DIR}/movielens100k.csv")

        msg = "Expected ratio to be of type float"
        with pytest.raises(TypeError,match= msg):
            split.random_split(data, wrong_dtype_ratio)

        msg = "Split ratio should be in between 0 and 1"
        with pytest.raises(ValueError,match= msg):
            split.random_split(data, out_of_bound_ratio)
        




