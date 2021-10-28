import pandas as pd
import numpy as np

from .eval_utils import merge_ranking_true_pred


def map_at_k(
    rating_true,
    rating_pred,
    col_user="userID",
    col_item="itemID",
    col_rating="ratings",
    col_prediction="predictions",
    relevancy_method="top_k",
    k=10,
    threshold=10,
):
    

    df_hit, df_hit_count, n_users = merge_ranking_true_pred(
        rating_true=rating_true,
        rating_pred=rating_pred,
        col_user=col_user,
        col_item=col_item,
        col_rating=col_rating,
        col_prediction=col_prediction,
        relevancy_method=relevancy_method,
        k=k,
        threshold=threshold,
    )

    if df_hit.shape[0] == 0:
        return 0.0

    df_hit_sorted = df_hit.copy()
    df_hit_sorted["recipr"] = (
        df_hit_sorted.groupby(col_user).cumcount() + 1
    ) / df_hit_sorted["rank"]
    df_hit_sorted = df_hit_sorted.groupby(col_user).agg({"recipr": "sum"}).reset_index()

    df_merge = pd.merge(df_hit_sorted, df_hit_count, on=col_user)
    return (df_merge["recipr"] / df_merge["actual"]).sum() / n_users




def precision_at_k(
    rating_true,
    rating_pred,
    col_user="userID",
    col_item="itemID",
    col_rating="ratings",
    col_prediction="predictions",
    relevancy_method="top_k",
    k=10,
    threshold=10,
):
    df_hit, df_hit_count, n_users = merge_ranking_true_pred(
        rating_true=rating_true,
        rating_pred=rating_pred,
        col_user=col_user,
        col_item=col_item,
        col_rating=col_rating,
        col_prediction=col_prediction,
        relevancy_method=relevancy_method,
        k=k,
        threshold=threshold,
    )

    if df_hit.shape[0] == 0:
        return 0.0

    return (df_hit_count["hit"] / k).sum() / n_users



def recall_k(
    rating_true,
    rating_pred,
    col_user="userID",
    col_item="itemID",
    col_rating="ratings",
    col_prediction="predictions",
    relevancy_method="top_k",
    k=10,
    threshold=10,
):
    df_hit, df_hit_count, n_users = merge_ranking_true_pred(
        rating_true=rating_true,
        rating_pred=rating_pred,
        col_user=col_user,
        col_item=col_item,
        col_rating=col_rating,
        col_prediction=col_prediction,
        relevancy_method=relevancy_method,
        k=k,
        threshold=threshold,
    )

    if df_hit.shape[0] == 0:
        return 0.0

    return (df_hit_count["hit"] / df_hit_count["actual"]).sum() / n_users