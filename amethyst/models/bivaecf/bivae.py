import itertools
from typing import ItemsView

import numpy as np
import torch
import torch.nn as nn

from tqdm.std import trange


EPS = 1e-10

ACT = {
    "sigmoid": nn.Sigmoid(),
    "tanh": nn.Tanh(),
    "elu": nn.ELU(),
    "relu": nn.ReLU(),
    "relu6": nn.ReLU6(),
}


class BiVAE(nn.Module):
    def __init__(self,
                k, 
                user_encoder_struc,
                item_encoder_structure,
                act_fn,
                likelihood,
                cap_priors,
                feature_dim,
                batch_size):
        super(BiVAE, self).__init__()

        self.mu_theta = torch.zeros((item_encoder_structure[0] * k))
        self.mu_beta = torch.zeros((user_encoder_struc[0] * k))

        self.theta = torch.randn(item_encoder_structure[0], k) * 0.01
        self.beta = torch.randn(user_encoder_struc[0], k) * 0.01
        torch.nn.init.kaiming_uniform_(self.theta, a=np.sqrt(5))

        self.likelihood = likelihood
        if act_fn not in ACT:
            raise ValueError(f"act_fn can be one of {ACT.keys()}")
        
        else:
            self.act_fn = ACT[act_fn]

        self.cap_priors = cap_priors
        if self.cap_priors.get("user", False):
            self.user_prior_encoder = nn.Linear(feature_dim.get("user"), k)
        if self.cap_priors.get("item", False):
            self.item_prior_encoder = nn.Linear(feature_dim.get("item"), k)

        
        self.user_encoder = nn.Sequential()
        for i in range(len(item_encoder_structure) - 1):
            self.user_encoder.add_module(
                f"fc{i}",
                nn.Linear(user_encoder_struc[i], user_encoder_struc[i+1])
            )
            self.user_encoder.add_module(f"act{i}", self.act_fn)
        
        self.user_mu = nn.Linear(user_encoder_struc[-1], k)
        self.user_std = nn.Linear(user_encoder_struc[-1], k)

        self.item_encoder = nn.Sequential()
        for i in range(len(item_encoder_structure) - 1):
            self.item_encoder.add_module(
                f"fc{i}",
                nn.Linear(item_encoder_structure[i], item_encoder_structure[i+1])
            )
            self.item_encoder.add_module(f"act{i}", self.act_fn)
        self.item_mu = nn.Linear(item_encoder_structure[-1], k)  # mu
        self.item_std = nn.Linear(item_encoder_structure[-1], k)