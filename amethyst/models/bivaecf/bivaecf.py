from re import VERBOSE
import numpy as np

from amethyst.dataloader.dataset import Dataloader

from ..base import BaseModel
from .bivae import BiVAE, train
import torch

class BiVAECF(BaseModel):
    def __init__(self, 
                name="BiVAECF",
                k=10,
                encoder_structure=[20],
                act_fn="tanh",
                likelihood="pois",
                n_epochs=100,
                batch_size=100,
                learning_rate=0.001,
                beta_kl=1.0,
                cap_priors={"user": False, "item": False},
                seed=None,
                use_gpu=True,
                trainable=True,
                verbose=False,
                ):
        BaseModel.__init__(self, name=name, trainable=trainable, verbose=verbose)

        self.k = k
        self.encoder_structure = encoder_structure
        self.act_fn = act_fn
        self.likelihood = likelihood
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.beta_kl = beta_kl
        self.cap_priors = cap_priors
        self.seed = seed
        self.use_gpu = use_gpu


    def fit(self, train_set: Dataloader, val_set=None):
        print(type(train_set))
        BaseModel.fit(self, train_set, val_set)

        self.device = (
            torch.device("cuda:0")
            if (self.use_gpu and torch.cuda.is_available())
            else torch.device("cpu")
        )

        if self.trainable:
            feature_dim = {
                "user": None,
                "item": None
            }

            if self.cap_priors.get("user", False):
                if train_set.user_feature is None:
                    raise ValueError("CAP priors for users is set to True but no user features were provided")

                else:
                    feature_dim["user"] = train_set.user_feature.feature_dim

            if self.cap_priors.get("item", False):
                if train_set.item_feature is None:
                    raise ValueError("CAP priors for items is set to True but no item features were provided")
                else:
                    feature_dim["item"] = train_set.item_feature.feature_dim

            if self.seed is not None:
                torch.manual_seed(self.seed)
                torch.cuda.manual_seed(self.seed)

            
            if not hasattr(self, 'bivaecf'):
                num_items = train_set.matrix.shape[1]
                num_users = train_set.matrix.shape[0]
                self.bivae = BiVAE(
                    k=self.k,
                    user_encoder_struc=[num_items] + self.encoder_structure,
                    item_encoder_struc=[num_users] + self.encoder_structure,
                    act_fn=self.act_fn,
                    likelihood=self.likelihood,
                    cap_priors=self.cap_priors,
                    feature_dim=feature_dim,
                    batch_size=self.batch_size,
                ).to(self.device)

            train(
                self.bivae,
                self.train_set,
                n_epochs=self.n_epochs,
                batch_size=self.batch_size,
                alpha=self.learning_rate,
                beta_kl=self.beta_kl,
                verbose=self.verbose,
                device=self.device,
            )

            
        elif self.verbose:
            print("Trainable is set to False")

        return self
        