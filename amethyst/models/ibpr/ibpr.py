import numpy as np
import torch

from ...dataloader.dataset import Dataloader
from tqdm.auto import tqdm

def base_impl(
    train_set: Dataloader,
    k,
    lambda_=0.001,
    epochs=150,
    alpha= 0.05,
    batch_size=100,
    init_params=None,
    verbose=False
):
    X = train_set.csr_matrix


    if init_params['U'] is None:
        u_factor = torch.randn(X.shape[0], k, requires_grad=True)

    else:
        u_factor = torch.from_numpy(init_params['U'])
        u_factor.requires_grad = True

    if init_params['V'] is None:
        v_factor = torch.randn(X.shape[1], k, requires_grad=True)

    else:
        v_factor = torch.from_numpy(init_params["V"])
        v_factor.requires_grad = True

    optim = torch.optim.Adam([u_factor, v_factor], lr=alpha)

    for epoch in range(1, epochs+1):
        sum_loss = 0.0
        ctr = 0

        progress_bar = tqdm(
            total=train_set.num_batches(batch_size),
            desc=f"Epoch {epoch}/{epochs}",
            disable= not verbose
        )

        for batch_u, batch_i, batch_j in train_set.uij_iter(batch_size):
            regU = u_factor[batch_u, :]
            regI = v_factor[batch_i, :]
            regJ = v_factor[batch_j, :]

            regU_unq = u_factor[np.unique(batch_u), :]
            regI_unq = v_factor[np.union1d(batch_i, batch_j), :]

            regU_norm = regU / regU.norm(dim=1)[:, None]
            regI_norm = regI / regI.norm(dim=1)[:, None]
            regJ_norm = regJ / regJ.norm(dim=1)[:, None]
        
            Scorei = torch.acos(
                torch.clamp(
                    torch.sum(regU_norm * regI_norm, dim=1), -1 + 1e-7, 1 - 1e-7
                )
            )
            Scorej = torch.acos(
                torch.clamp(
                    torch.sum(regU_norm * regJ_norm, dim=1), -1 + 1e-7, 1 - 1e-7
                )
            )

            loss = (
                lambda_ * (regU_unq.norm().pow(2) + regI_unq.norm().pow(2))
                - torch.log(torch.sigmoid(Scorej - Scorei)).sum()
            )
            optim.zero_grad()
            loss.backward()
            optim.step()

            sum_loss += loss.data.item()
            ctr += len(batch_u)
            if ctr % (batch_size * 10) == 0:
                progress_bar.set_postfix(loss=(sum_loss / ctr))
            progress_bar.update(1)

        progress_bar.close()


    u_factor = torch.nn.functional.normalize(u_factor, p=2, dim=1)
    v_factor = torch.nn.functional.normalize(v_factor, p=2, dim=1)
    u_factor = u_factor.data.cpu().numpy()
    v_factor = v_factor.data.cpu().numpy()

    res = {"U": u_factor, "V": v_factor}

    return res