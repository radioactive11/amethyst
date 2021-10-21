from numpy.core.numeric import outer
import pandas as pd
import numpy as np

from .base import BaseModel
from ..dataloader.dataset import Dataloader


def rank(model: BaseModel, data, user_col="user", item_col="item", rank_col="rank", remove_seen=False):
    users, items, preds = [], [], []

    item = model.train_set.item_id_mapping.keys()

    for user_id, user_idx, in model.train_set.user_id_mapping.items():
        user = [user_id] * len(item)
        users.extend(user)
        items.extend(item)
        preds.extend(model.eval(user_idx).tolist())

    all_preds = pd.DataFrame(
        data = {
            user_col: users,
            item_col: items,
            rank_col: preds
        }
    )

    if remove_seen:
        temp_df = pd.concat(
            [
                data[[user_col, item_col]],
                pd.DataFrame(
                    data=np.ones(data.shape[0]), columns=["dummycol"], index=data.index
                ),
            ],
            axis=1,
        )
        merged = pd.merge(temp_df, all_preds, on=[user_col, item_col], how="outer")
        return merged[merged["dummycol"].isnull()].drop("dummycol", axis=1)

    else:
        return all_preds

