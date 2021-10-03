from collections import defaultdict, OrderedDict


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
        self.__user_data = None
        self.__item_data = None

    
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
        """Set total users for dataset"""
        assert input_val >= self.user_count
        self.__total_users = input_val


    @property
    def total_items(self) -> int:
        """Total number of items"""
        return self.__total_items if self.__total_items is not None else self.item_count

    @total_items.setter
    def total_items(self, input_val):
        """Sets total number of items for dataset"""
        assert input_val >= self.num_items
        self.__total_items = input_val


    @property
    def user_ids(self):
        """Iterator for user ids"""
        return self.user_id_mapping.keys()


    @property
    def item_ids(self):
        """Iterator for item ids"""
        return self.item_id_mapping.keys()

    
    @property
    def user_indices(self):
        """Iterator for user indices"""
        return self.user_id_mapping.values()

    @property
    def item_indices(self):
        """Iterator for item indices"""
        return self.item_id_mapping.values()

    
    @property
    def user_data(self):
        if self.__user_data is not None:
            self.__user_data = defaultdict()
            for user, item, rating in zip(*self.user_item_rating):
                __u_data = self.__user_data.setdefault(user, ([], []))
                __u_data[0].append(item)
                __u_data[1].append(rating)

            return self.__user_data
        

    @classmethod
    def build(cls, data, global_user_map=None, global_item_map=None):
        if global_user_map is None:
            global_user_map = OrderedDict()
        
        if global_item_map is None:
            global_item_map = OrderedDict()

        user_id_map = OrderedDict()
        item_id_map = OrderedDict()

        user_indices = []
        item_indices = []
        ratings = []
        valid_idx = []

        user_index_set = set()
        duplicates = 0

        for idx, (user_id, item_id, rating, *_)


