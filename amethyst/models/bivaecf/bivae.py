import itertools

import numpy as np
import torch
from torch._C import Value
import torch.nn as nn
from torch.nn import parameter

from tqdm.std import trange

from dataloader.dataset import Dataloader

# std deviation 
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
                item_encoder_struc,
                act_fn,
                likelihood,
                cap_priors,
                feature_dim,
                batch_size):
        super(BiVAE, self).__init__()

        self.mu_theta = torch.zeros((item_encoder_struc[0], k))
        self.mu_beta = torch.zeros((user_encoder_struc[0], k))

        self.theta = torch.randn(item_encoder_struc[0], k) * 0.01
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
        for i in range(len(item_encoder_struc) - 1):
            self.user_encoder.add_module(
                f"fc{i}",
                nn.Linear(user_encoder_struc[i], user_encoder_struc[i+1])
            )
            self.user_encoder.add_module(f"act{i}", self.act_fn)
        
        self.user_mu = nn.Linear(user_encoder_struc[-1], k)
        self.user_std = nn.Linear(user_encoder_struc[-1], k)

        self.item_encoder = nn.Sequential()
        for i in range(len(item_encoder_struc) - 1):
            self.item_encoder.add_module(
                f"fc{i}",
                nn.Linear(item_encoder_struc[i], item_encoder_struc[i+1])
            )
            self.item_encoder.add_module(f"act{i}", self.act_fn)
        self.item_mu = nn.Linear(item_encoder_struc[-1], k)  # mu
        self.item_std = nn.Linear(item_encoder_struc[-1], k)


    def to(self, device):
        self.beta = self.beta.to(device=device)
        self.theta = self.theta.to(device=device)
        self.mu_beta = self.mu_beta.to(device=device)
        self.mu_theta = self.mu_theta.to(device=device)
        return super(BiVAE, self).to(device)

    
    def encode_user_prior(self, pre):
        result = self.user_prior_encoder(pre)
        return result


    def encode_user_prior(self, pre):
        result = self.item_prior_encoder(pre)
        return result


    def encode_user(self, x):
        u = self.user_encoder(x)
        return self.user_mu(u), torch.sigmoid(self.user_std(u))


    def encode_item(self, x):
        i = self.item_encoder(x)
        return self.user_mu(i), torch.sigmoid(self.user_std(i))

    
    def decode_user(self, theta, beta):
        u_ = theta.mm(beta.t()) # matmul
        return torch.sigmoid(u_)

    
    def decode_item(self, theta, beta):
        i_ = beta.mm(theta.t())
        return torch.sigmoid(i_)

    
    def reparameterize(self, mu, std):
        # z = mu + sigma • epsilon
        eps = torch.randn_like(mu)
        return mu + eps * std

    
    def forward(self, x, user=True, beta=None, theta=None):
        if user:
            mu, std = self.encode_user(x)
            theta = self.reparameterize(mu, std)
            return theta, self.decode_user(theta, beta), mu, std

        else:
            mu, std = self.encode_item(x)
            beta = self.reparameterize(mu, std)
            return beta, self.decode_item(theta, beta), mu, std

    
    def loss(self, x, x_, mu, mu_prior, std, kl_beta):
        ll_choices = {
            "bern": x * torch.log(x_ + EPS) + (1 - x) * torch.log(1 - x_ + EPS),
            "gaus": -(x - x_) ** 2,
            "pois": x * torch.log(x_ + EPS) - x_,
        }

        if self.likelihood not in ll_choices:
            raise ValueError(f"likelihood can take values from {ll_choices.keys()} only")
        
        loss_fn = ll_choices[self.likelihood]

        loss_fn = torch.sum(loss_fn, dim=1)

        kld = -0.5 * (1 + 2.0 * torch.log(std) - (mu - mu_prior).pow(2) - std.pow(2))
        kld = torch.sum(kld, dim=1)

        return torch.mean(kl_beta * kld - loss_fn)


def train(
    bivae: BiVAE,
    train_set,
    n_epochs,
    batch_size,
    alpha,
    beta_kl,
    verbose,
    device=torch.device("cpu"),
    dtype=torch.float32
        ):
    user_params = itertools.chain(
        bivae.user_encoder.parameters(),
        bivae.user_mu.parameters(),
        bivae.user_std.parameters()
    )

    item_params = itertools.chain(
        bivae.item_encoder.parameters(),
        bivae.item_mu.parameters(),
        bivae.item_std.parameters()
    )
    
    u_optim = torch.optim.Adam(params=user_params, lr=alpha)
    i_optim = torch.optim.Adam(params=item_params, lr=alpha)

    X = train_set.matrix.copy()

    X.data = np.ones_like(X.data)

    tx = X.transpose()

    progress_bar = trange(1, n_epochs+1, disable=not verbose)

    for _ in progress_bar:
        i_sum_loss = 0.0
        i_count = 0

        for i_ids in train_set.item_iter(batch_size):
            i_batch = tx[i_ids, :]
            i_batch = i_batch.A
            i_batch = torch.tensor(i_batch, dtype=dtype, device=device)

            beta, i_batch_, i_mu, i_std = bivae(i_batch, user=False, theta=bivae.theta)

            i_mu_prior = 0.0  

            i_loss = bivae.loss(i_batch, i_batch_, i_mu, i_mu_prior, i_std, beta_kl)
            i_optim.zero_grad()
            i_loss.backward()
            i_optim.step()

            i_sum_loss += i_loss.data.item()
            i_count += len(i_batch)

            beta, _, i_mu, _ = bivae(i_batch, user=False, theta=bivae.theta)
            
            bivae.beta.data[i_ids] = beta.data
            bivae.mu_beta.data[i_ids] = i_mu.data


        u_sum_loss = float(0)
        u_count = 0
        for u_ids in train_set.user_iter(batch_size):
            u_batch = X[u_ids, :]
            u_batch = u_batch.A
            u_batch = torch.tensor(u_batch, dtype=dtype, device=device)

            theta, u_batch_, u_mu, u_std = bivae(u_batch, user=True, beta=bivae.beta)

            u_mu_prior = float(0)

            u_loss = bivae.loss(u_batch, u_batch_, u_mu, u_mu_prior, u_std, beta_kl)
            u_optim.zero_grad()
            u_loss.backward()
            u_optim.step()

            u_sum_loss += u_loss.data.item()
            u_count += len(u_batch)

            theta, _, u_mu, _ = bivae(u_batch, user=True, beta=bivae.beta)
            bivae.theta.data[u_ids] = theta.data
            bivae.mu_theta.data[u_ids] = u_mu.data

            progress_bar.set_postfix(
                loss_i=(i_sum_loss / i_count), loss_u=(u_sum_loss / (u_count))
            )

    for i_ids in train_set.item_iter(batch_size):
        i_batch = tx[i_ids, :]
        i_batch = i_batch.A
        i_batch = torch.tensor(i_batch, dtype=dtype, device=device)

        beta, _, i_mu, _ = bivae(i_batch, user=False, theta=bivae.theta)
        bivae.mu_beta.data[i_ids] = i_mu.data

    for u_ids in train_set.user_iter(batch_size):
        u_batch = X[u_ids, :]
        u_batch = u_batch.A
        u_batch = torch.tensor(u_batch, dtype=dtype, device=device)

        theta, _, u_mu, _ = bivae(u_batch, user=True, beta=bivae.beta)
        bivae.mu_theta.data[u_ids] = u_mu.data

    return bivae

        
