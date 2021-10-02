import numpy as np
import inspect
import copy
import datetime
import os
import pickle

from numpy.lib.npyio import save


class BaseModel():
    def __init__(self, name: str, trainable: True, verbose=False) -> None:
        self.name = name
        self.trainable = trainable
        self.verbose = verbose
        
        self.train_set = None
        self.test_set = None

        self.ignore = [self.train_set, self.test_set]

    
    def reset(self):
        self.best_value = -np.Inf
        self.best_epoch = 0
        self.current_epoch = 0
        self.stopped_epoch = 0
        self.wait = 0

    def __deeepcopy__(self, memo):
        """Copy properties without changing the original object

        Args:
            memo (dict): maintaining already copied props while recurring

        Returns:
            None
        """

        #* https://docs.python.org/3/library/copy.html
        obj = self.__class_s_
        result = obj.__new__(obj)

        for prop, val in self.__dict__.items():
            if prop == self.train_set or prop == self.test_set:
                continue

            setattr(result, prop, copy.deepcopy(val))
        return result


    @classmethod
    def _get_init_params(obj):
        init = getattr(obj.__init__, "deprecated_original", obj.__init__)
        if init is object.__init__:
            return []

        init_signature = inspect.signature(init)
        parameters = [p for p in init_signature.parameters.values() if p.name != 'self']
        return sorted([p.name for p in parameters])


    def save(self, model_name=None):
        SAVE_DIR = os.path.join(os.getcwd(), "saves")
        os.makedirs(SAVE_DIR, exist_ok=True)
        filename = model_name if model_name is not None else self.name

        filepath = os.path.join(SAVE_DIR, filename)

        save_model = copy.deepcopy(self)

        pickle.dump(
            save_model, open(filepath, "wb"), protocol=pickle.HIGHEST_PROTOCOL
        )

        if self.verbose:
            print(f"[SAVED] {self.name} model saved to ")





