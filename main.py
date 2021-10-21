from amethyst.dataloader import split, dataset
import pandas as pd
import torch
from amethyst.models.bivaecf.bivaecf import BiVAECF

from amethyst.models.predictions import rank


df = pd.read_csv("./data/movielens100k.csv")

train, test = split.stratified_split(df,
                                    0.8, 
                                    user_col='userID',
                                    item_col='itemID',
                                    filter_col='item'
                                    )


X = dataset.Dataloader.dataloader(train.itertuples(index=False))
y = dataset.Dataloader.dataloader(test.itertuples(index=False))

print('Number of users: {}'.format(X.user_count))
print('Number of items: {}'.format(X.item_count))

LATENT_DIM = 50
ENCODER_DIMS = [100]
ACT_FUNC = "tanh"
LIKELIHOOD = "pois"
NUM_EPOCHS = 500
BATCH_SIZE = 128
LEARNING_RATE = 0.001

bivae = BiVAECF(
    k=LATENT_DIM,
    encoder_structure=ENCODER_DIMS,
    act_fn=ACT_FUNC,
    likelihood=LIKELIHOOD,
    n_epochs=NUM_EPOCHS,
    batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    seed=42,
    use_gpu=torch.cuda.is_available(),
    verbose=True
)

mdl = bivae.fit(X, y)

all_preds = rank(mdl, y, user_col='userID', item_col='itemID')

all_preds.to_csv("predictions.csv", index=False)