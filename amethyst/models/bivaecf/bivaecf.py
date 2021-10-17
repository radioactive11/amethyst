import numpy as np

from ..base import BaseModel



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
        self.likelihood = likelihood
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.beta_kl = beta_kl
        self.cap_priors = cap_priors
        self.seed = seed
        self.use_gpu = use_gpu


    def fit(self, train_set, test_set=None):
        BaseModel.fit(train_set, test_set)
        