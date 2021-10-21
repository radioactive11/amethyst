from collections import defaultdict, OrderedDict
from typing import Counter
import warnings

import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix, dok_matrix

from .data_utils import estimate_batches, get_rng

class Dataloader(object):
    def __init__(
                self,
                user_count,
                item_count,
                user_id_mapping,
                item_id_mapping,
                user_item_rating,
                seed=None):
        self.user_count = user_count
        self.item_count = item_count
        self.user_id_mapping = user_id_mapping
        self.item_id_mapping = item_id_mapping
        self.user_item_rating = user_item_rating
        self.rng_seed = seed
        self.rng = get_rng(seed)

        (_, _, rating_vals) = user_item_rating
        self.ratings_count = len(rating_vals)
        self.max_rating_val = np.max(rating_vals)
        self.min_rating_val = np.min(rating_vals)
        self.global_rating_mean = np.mean(rating_vals)

        
        self.__total_users = None
        self.__total_items = None
        self.__user_data = None
        self.__item_data = None

        self.__csr_matrix = None
        self.__dok_matrix = None
    
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
    

    @property
    def item_data(self):
        if self.__item_data is None:
            self.__item_data = defaultdict()
            for u, i, r in zip(*self.user_index_rating_tuple):
                i_data = self.__item_data.setdefault(i, ([], []))
                i_data[0].append(u)
                i_data[1].append(r)
        return self.__item_data


    @property
    def matrix(self):
        return self.csr_matrix


    @property
    def csr_matrix(self):
        if self.__csr_matrix is None:
            (u_indices, i_indices, r_values) = self.user_item_rating
            self.__csr_matrix = csr_matrix(
                (r_values, (u_indices, i_indices)),
                shape=(self.user_count, self.item_count),
            )
        return self.__csr_matrix

    # spase matric as dict of key
    @property
    def dok_matrix(self):
        """The user-item interaction matrix in DOK sparse format"""
        if self.__dok_matrix is None:
            self.__dok_matrix = dok_matrix(
                (self.user_count, self.item_count), dtype=np.float32
            )
            for u, i, r in zip(*self.user_item_rating):
                self.__dok_matrix[u, i] = r
        return self.__dok_matrix


    @classmethod
    def build(cls, data, exclude_unknown=False):
        global_user_map = OrderedDict()
        global_item_map = OrderedDict()
        user_id_map = OrderedDict()
        item_id_map = OrderedDict()

        user_indices = []
        item_indices = []
        ratings = []
        valid_idx = []

        user_index_set = set()
        duplicates = 0

        for idx, (_user_id, _item_id, rating, *_) in enumerate(data):
            # exclude duplicates if  set to true
            if exclude_unknown and \
            (_user_id not in global_user_map or _item_id not in global_item_map):
                continue
            
            if (_user_id, _item_id) in user_index_set:
                duplicates += 1
                continue

            user_index_set.add((_user_id, _item_id))

            user_id_map[_user_id] = global_user_map.setdefault(_user_id, len(global_user_map))
            item_id_map[_item_id] = global_item_map.setdefault(_item_id, len(global_item_map))

            # save valid user, items, ratings and indices
            user_indices.append(user_id_map[_user_id])
            item_indices.append(item_id_map[_item_id])
            ratings.append(float(rating))
            valid_idx.append(idx)

        if duplicates > 0:
            warnings.warn(f"{duplicates} duplicates removed.")
        
        if len(user_index_set) == 0:
            raise ValueError("Empty dataset after de-dup")

        user_index_rating_tuple = (
            np.asarray(user_indices, dtype=np.int),
            np.asarray(item_indices, dtype=np.int),
            np.asarray(ratings, dtype=np.float)
        )

        return cls(
            user_count=len(global_user_map),
            item_count=len(global_item_map),
            user_id_mapping=user_id_map,
            item_id_mapping=item_id_map,
            user_item_rating = user_index_rating_tuple
        )

        
    @classmethod
    #* this is from uir
    def dataloader(cls, data, seed=None):
        return cls.build(data)

    def num_batches(self, batch_size):
        sample_length = len(self.user_item_rating[0])
        return estimate_batches(sample_length, batch_size)


    def idx_iter(self, idx_range, batch_size=1):
        indices = np.arange(idx_range)

        n_batches = estimate_batches(len(indices), batch_size)
        
        for batch in range(n_batches):
            start_offset = batch_size * batch
            end_offset = batch_size * batch + batch_size
            # check for out of bound
            end_offset = min(end_offset, len(indices))
            batch_ids = indices[start_offset:end_offset]
            yield batch_ids


    def uir_iter(self, batch_size=1, binary=False, num_zeros=0):
        for batch_ids in self.idx_iter(len(self.user_item_rating[0]), batch_size):
            batch_users = self.user_item_rating[0][batch_ids]
            batch_items = self.user_item_rating[1][batch_ids]
            if binary:
                batch_ratings = np.ones_like(batch_items)
            else:
                batch_ratings = self.user_item_rating[2][batch_ids]

            if num_zeros > 0:
                repeated_users = batch_users.repeat(num_zeros)
                neg_items = np.empty_like(repeated_users)
                for i, u in enumerate(repeated_users):
                    j = self.rng.randint(0, self.num_items)
                    while self.dok_matrix[u, j] > 0:
                        j = self.rng.randint(0, self.num_items)
                    neg_items[i] = j
                batch_users = np.concatenate((batch_users, repeated_users))
                batch_items = np.concatenate((batch_items, neg_items))
                batch_ratings = np.concatenate(
                    (batch_ratings, np.zeros_like(neg_items))
                )

            yield batch_users, batch_items, batch_ratings


    def uij_iter(self, batch_size=1, neg_sampling="uniform"):
        if neg_sampling == "uniform":
            neg_population = np.arange(self.item_count)

        elif neg_sampling == "popularity":
            neg_population = self.user_item_rating[1]
        
        else:
            raise ValueError(f"Invalid sampling option {neg_sampling}")
        

        for batch_ids in self.idx_iter(len(self.user_item_rating[0]), batch_size):
            batch_users = self.user_item_rating[0][batch_ids]
            batch_pos_items = self.user_item_rating[1][batch_ids]
            batch_pos_ratings = self.user_item_rating[2][batch_ids]
            batch_neg_items = np.empty_like(batch_pos_items)
            for i, (user, pos_rating) in enumerate(zip(batch_users, batch_pos_ratings)):
                neg_item = self.rng.choice(neg_population)
                while self.dok_matrix[user, neg_item] >= pos_rating:
                    neg_item = self.rng.choice(neg_population)
                batch_neg_items[i] = neg_item
            yield batch_users, batch_pos_items, batch_neg_items


    def user_iter(self, batch_size=1):
        user_indices = np.fromiter(self.user_indices, dtype=np.int)
        for batch_ids in self.idx_iter(len(user_indices), batch_size):
            yield user_indices[batch_ids]


    def item_iter(self, batch_size=1):
        item_indices = np.fromiter(self.item_indices, np.int)
        for batch_ids in self.idx_iter(len(item_indices), batch_size):
            yield item_indices[batch_ids]


    def is_unknown_user(self, user_idx):
        return user_idx >= self.user_count


    def is_unknown_item(self, item_idx):
        return item_idx >= self.item_count



