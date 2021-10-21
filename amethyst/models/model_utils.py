import numpy as np
from numpy.lib.utils import source


def clip(values, upper_bound, lower_bound):
    values = np.where(values > upper_bound, upper_bound, values)
    values = np.where(values < lower_bound, lower_bound, values)

    return values


def scale(values, target_min, target_max, source_min=None, source_max=None):
    if source_min is None:
        source_min = np.min(values)

    if source_max is None:
        source_max = np.max(values)

    if source_max == source_min:
        source_min = 0.0

    values = (values - source_min) / (source_max - source_min)
    values = values * (target_max - target_min) + target_min
    return values

