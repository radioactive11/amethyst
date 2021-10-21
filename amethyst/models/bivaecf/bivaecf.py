from torch._C import Value
from amethyst.dataloader.dataset import Dataloader

from .bivae import BiVAE, train
from ..base import BaseModel
from ..model_utils import scale
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
        
    
    def eval(self, user_idx, item_idx=None):
        if item_idx is None:
            if self.train_set.is_unknown_user(user_idx):
                raise ValueError(
                    "Can't make score prediction for (user_id=%d)" % user_idx
                )

            theta_u = self.bivae.mu_theta[user_idx].view(1, -1)
            beta = self.bivae.mu_beta
            known_item_scores = (
                self.bivae.decode_user(theta_u, beta).cpu().numpy().ravel()
            )

            return known_item_scores
        else:
            if self.train_set.is_unknown_user(user_idx) or self.train_set.is_unknown_item(
                item_idx
            ):
                raise ValueError(
                    "Can't make score prediction for (user_id=%d, item_id=%d)"
                    % (user_idx, item_idx)
                )

            theta_u = self.bivae.mu_theta[user_idx].view(1, -1)
            beta_i = self.bivae.mu_beta[item_idx].view(1, -1)
            pred = self.bivae.decode_user(theta_u, beta_i).cpu().numpy().ravel()

            pred = scale(
                pred, self.train_set.min_rating, self.train_set.max_rating, 0.0, 1.0
            )

            return pred


            