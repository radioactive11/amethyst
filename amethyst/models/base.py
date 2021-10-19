import numpy as np
import inspect
import copy
import datetime
import os
import pickle

from torch._C import import_ir_module

from amethyst.models.bivaecf.bivae import train

from .model_utils import clip



class BaseModel():
    def __init__(self, name: str, trainable=True, verbose=False) -> None:
        self.name = name
        self.trainable = trainable
        self.verbose = verbose
        
        self.train_set = None
        self.test_set = None

        self.ignore = [self.train_set, self.test_set]
        self._secret_stuff = 0
    
    def reset(self):
        self.best_value = -np.Inf
        self.best_epoch = 0
        self.current_epoch = 0
        self.stopped_epoch = 0
        self.wait = 0

    def __deeepcopy__(self, memo):
        """Copy properties without changing the original clsect

        Args:
            memo (dict): maintaining already copied props while recurring

        Returns:
            None
        """

        #* https://docs.python.org/3/library/copy.html
        cls = self.__class_s_
        result = cls.__new__(cls)

        for prop, val in self.__dict__.items():
            if prop == self.train_set or prop == self.test_set:
                continue

            setattr(result, prop, copy.deepcopy(val))
        return result


    @classmethod
    def _get_init_params(cls):
        init = getattr(cls.__init__, "deprecated_original", cls.__init__)
        if init is object.__init__:
            return []

        init_signature = inspect.signature(init)
        parameters = [p for p in init_signature.parameters.values() if p.name != 'self']
        return sorted([p.name for p in parameters])


    def save(self, model_name: str = None):
        """Save trained model

        Args:
            model_name (str, optional): Filename by which model is to be saved. Defaults to None.

        Returns:
            str: filename of the model
        """
        SAVE_DIR = os.path.join(os.getcwd(), "saves")
        os.makedirs(SAVE_DIR, exist_ok=True)
        filename = model_name if model_name is not None else self.name
        filepath = os.path.join(SAVE_DIR, filename)
        save_model = copy.deepcopy(self)

        pickle.dump(save_model, open(filepath, "wb"), protocol=pickle.HIGHEST_PROTOCOL)

        if self.verbose:
            print(f"[SAVED] {self.name} model saved to {SAVE_DIR}")
        
        return filepath


    @staticmethod
    def load(model_path, trainable: bool=False):
        # Todo Complete this
        """Load model from pickled file

        Args:
            model_path (str): path to model file
            trainable (bool, optional): Whether to stage model for training. Defaults to False.

        Raises:
            FileNotFoundError: [description]

        Returns:
            [type]: [description]
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"File not found: {model_path}")
        
        file_bytes = open(model_path, 'rb')
        model = pickle.load(file_bytes)
        model.trainable = trainable
        model.load_from = model_path

        return model


    def fit(self, train_set, val_set = None):
        self.train_set = train_set
        self.test_set = val_set
        return self

    
    def predict(self, user_idx, item_idx=None):
        #! score
        """Predict the rating of a user for an item.
        Args:
            user_idx (int): user index
            item_idx (int, optional): Item for which rating is to be calculated. All items
            are considered if None. Defaults to None.
        """
        raise NotImplementedError("This model cannot predict yet")


    def default_predictions(self):
        return self.train_set.global_rating_mean


    def monitor_value(self):
        raise NotImplementedError()


    def rate(self, user_idx, item_idx, clipping: True):
        """Assign a rating score for a user and irem

        Args:
            user_idx (int): user ID
            item_idx (int): item ID
            clipping (Bool): whether to limit the rating

        Returns:
            int: score for user item pair
        """
        try:
            ratings = self.predict(user_idx, item_idx)
        
        except:
            ratings = self.default_predictions()

        if clipping:
            clip(
                values=ratings, 
                upper_bound=self.train_set.max_rating_val, 
                lower_bound=self.test_set.min_rating_val
            )

        return ratings


    def early_stop(self, min_delta=0.0, patience=0):
        self.current_epoch += 1
        
        current_value = self.monitor_value()
        if current_value is None:
            return False

        if np.greater_equal(current_value - self.best_value, min_delta):
            self.best_value = current_value
            self.best_epoch = self.current_epoch
            self.wait = 0

        else:
            self.wait += 1
            if self.wait >= patience:
                self.stopped_epoch = self.current_epoch

        if self.stopped_epoch > 0:
            print("Early Stopping Triggered")
            print(f"--Best Epoch: {self.best_epoch} | Stopped at Epoch {self.stopped_epoch} --")
            print(f"--Best Monitored Value: {self.best_value} | Delta: {current_value - self.best_value}")

            return True
        return False




