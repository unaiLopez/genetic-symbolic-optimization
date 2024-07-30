import numpy as np

from typing import Callable

class Loss:
    def _calculate_mae(self, y: np.array, y_pred: np.array) -> float:
        return np.mean(np.abs(y - y_pred))
    
    def _calculate_mse(self, y: np.array, y_pred: np.array) -> float:
        return np.mean((y - y_pred) ** 2)
    
    def _calculate_rmse(self, y: np.array, y_pred: np.array) -> float:
        return np.mean(np.sqrt(np.mean((y - y_pred) ** 2)))
    
    def get_loss_function(self, loss_type: str) -> Callable:
        if loss_type  == "mae":
            return self._calculate_mae
        elif loss_type  == "mse":
            return self._calculate_mse
        elif loss_type  == "rmse":
            return self._calculate_rmse
        else:
            raise NotImplementedError(f"Loss function {self.loss_type } is not implemented...")
