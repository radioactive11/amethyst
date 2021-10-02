from _typeshed import Self
from functools import total_ordering
import numpy as np
import pandas as pd



class Dataloader(object):
    def __init__(
                self,
                user_count,
                item_count,
                user_id_mapping,
                item_id_mapping,
                user_item_rating,
                seed=None,
    ):
        self.user_count = user_count
        self.item_count = item_count
        self.user_id_mapping = user_id_mapping
        self.item_id_mapping = item_id_mapping
        self.user_item_rating = user_item_rating
        self.rng_seed = seed

        (_, _, rating_vals) = user_item_rating
        self.ratings_count = len(rating_vals)
        self.max_rating_val = np.max(rating_vals)
        self.min_rating_val = np.min(rating_vals)
        self.global_rating_mean = np.mean(rating_vals)

        
        self.__total_users = None
        self.__total_items = None
        
    
    #* Propoerties

    @property
    def total_users(self) -> int:
        """Return total number of users

        Returns:
            int: total number of users
        """
        return self.__total_users if self.__total_users is not None else self.user_count

    @total_users.setter
    def total_users(self, input_val):
        assert input_val >= self.user_count
        self.__total_users = input_val


    @property
    def total_items(self) -> int:
        return self.__total_items if self.__total_items is not None else self.item_count

    @total_items.setter
    def total_items(self, input_val):
        assert input_val >= self.num_items
        self.__total_items = input_val

    


        