from posixpath import split
from typing import Optional
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from dataloader.data_utils import verify_split_ratio, verify_stratification, ratio_split

import os

def random_split(data: pd.DataFrame, split_ratio: float, seed: Optional[int]=11):
    if not isinstance(seed, int):
        raise TypeError("Expected seed to be of type int")

    ratio = verify_split_ratio(split_ratio)

    return train_test_split(data, test_size=None, train_size=ratio, random_state=seed)


def stratified_split(
    data: pd.DataFrame,
    ratio: float,
    user_col: str,
    item_col: str,
    filter_col: str,
    min_rating_count: Optional[int] = 1,
    seed: Optional[int] = 11
):
    """Stratified split according to ratio

    Args:
        data (pd.DataFrame): data to be split
        ratio (float): ratio in which data is to be split
        user_col (str): column name containing user_ids
        item_col (str): column name containing item_ids
        filter_col (str): Either user or item
        min_rating (Optional[int], optional): Minimum items a user has rated before considered eligible for scoring. Defaults to 1.
        seed (Optional[int], optional): Seed for RNG. Defaults to 11.
    """
    verify_stratification(data, ratio, user_col, item_col, filter_col, min_rating_count, seed)
    split_col = user_col if filter_col == 'user' else item_col
    ratio_list = [ratio, 1-ratio]

    # consider only those users/items who ratings more than `min_rating_count`

    if min_rating_count > 1: data = data.groupby(split_col).filter(lambda x: len(x) >= min_rating_count)
    groups = data.groupby(split_col) # iterable of dataframes

    splits = []

    for _, group in groups:
        split_by_group = ratio_split(group, ratio_list)

        group_splits = pd.concat(split_by_group) 

        splits.append(group_splits)

    all_splits = pd.concat(splits)

    splits_list = [
        all_splits[all_splits["split_index"] == x].drop("split_index", axis=1)
        for x in range(len(ratio_list))
    ]

    return splits_list
    

if __name__ == "__main__":
    BASE_DIR = os.getcwd()
    DATA_DIR = os.path.join(BASE_DIR, "data")
    df = pd.read_csv(f"{DATA_DIR}/movielens100k.csv")
    # train, test = random_split(df, 0.75, 11)
    train, test = stratified_split(df, 0.8, 'userID', 'itemID', 'item')
    train.to_csv("train.csv", index=False)

