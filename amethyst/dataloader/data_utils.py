import pandas as pd
import numpy as np
import math



def verify_split_ratio(ratio: float) -> float:
    """Verifies if ratio for split is acceptable (float in range(0, 1)).

    Args:
        ratio (float): ratio in which data is to be split

    Returns:
        ratio if value is acceptable, else raises ValueError
    """
    if not isinstance(ratio, float):
        raise ValueError("Expected ratio to be of type float")

    if ratio <= 0 or ratio >= 1:
        raise ValueError("Split ratio should be in between 0 and 1")
    
    return ratio