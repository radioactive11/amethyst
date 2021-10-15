import numpy as np



def clip(values, upper_bound, lower_bound):
    values = np.where(values > upper_bound, upper_bound, values)
    values = np.where(values < lower_bound, lower_bound, values)

    return values