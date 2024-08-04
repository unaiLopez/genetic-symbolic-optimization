import numpy as np

from typing import Callable

def calculate_mae(y: np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean(np.abs(y - y_pred))
    
def calculate_mse(y: np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean((y - y_pred) ** 2)

def calculate_rmse(y: np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean(np.sqrt(np.mean((y - y_pred) ** 2)))

def get_loss_function(loss_type: str) -> Callable:
    if loss_type  == "mae":
        return calculate_mae
    elif loss_type  == "mse":
        return calculate_mse
    elif loss_type  == "rmse":
        return calculate_rmse
    else:
        raise NotImplementedError(f"Loss function {loss_type} is not implemented...")