import numpy as np

from typing import Callable

class Score:
    def _calculate_r_squared(self, y: np.array, y_pred: np.array) -> float:
        y_mean = np.mean(y)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y_mean) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        return r2 
    
    def get_score_function(self, score_type: str) -> Callable:
        if score_type  == "r2":
            return self._calculate_r_squared
        else:
            raise NotImplementedError(f"Score function {self.score_type} is not implemented...")
    