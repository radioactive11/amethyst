from typing import ItemsView
import numpy as np
from torch._C import Value

from amethyst.dataloader.dataset import Dataloader
from amethyst.models.bivaecf.bivae import train

from .ibpr import base_impl
from ..base import BaseModel


class IBPR(BaseModel):
    def __init__(
        self,
        k=20,
        max_iter=100,
        alpha_=0.05,
        lambda_=0.001,
        batch_size=100,
        name="IBPR",
        trainable=True,
        verbose=False,
        init_params=None):
        BaseModel.__init__(self, name=name, trainable=trainable, verbose=verbose)

        self.k = k
        self.max_iter = max_iter
        self.name = name
        self.alpha_ = alpha_
        self.lambda_  = lambda_
        self.batch_size = batch_size

        self.init_params = {} if init_params is None else init_params
        self.U = self.init_params.get("U", None)
        self.V = self.init_params.get("V", None)


    def fit(self, train_set: Dataloader, val_set=None):
        self.train_set = train_set
        self.val_set = val_set
        if self.trainable:
            res = base_impl(
                self.train_set,
                k=self.k,
                epochs=self.max_iter,
                lambda_=self.lambda_,
                alpha=self.alpha_,
                batch_size=self.batch_size,
                init_params={"U": self.U, "V": self.V},
                verbose=self.verbose,
            )
            self.U = np.asarray(res["U"])
            self.V = np.asarray(res["V"])

        return self

    
    def eval(self, user_idx, item_idx=None):
        if item_idx is None:
            if self.train_set.is_unknown_user(user_idx):
                raise ValueError(f"Cannot make predictions for userID {user_idx}")
            
            item_scores = self.V.dot(self.U[user_idx, :])
            return item_scores

        else:
            if self.train_set.is_unknown_user(user_idx) or self.train_set.is_unknown_user(user_idx):
                raise ValueError(f"Cannot make predictions for userID {user_idx} & itemID {item_idx}")

            user_pred = self.V[item_idx, :].dot(self.U[user_idx, :])
            return user_pred
            

