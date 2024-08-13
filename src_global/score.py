import numpy as np

from typing import Callable

def calculate_r2(y: np.ndarray, y_pred: np.ndarray) -> float:
    y_mean = np.mean(y)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y_mean) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    return r2           #return np.clip(r2, -1.0, 1.0)

def get_score_function(score_type: str) -> Callable:
    if score_type  == "r2":
        return calculate_r2
    else:
        raise NotImplementedError(f"Score function {score_type} is not implemented...")
