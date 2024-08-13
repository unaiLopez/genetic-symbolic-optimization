import numpy as np

from typing import Tuple

class AdamOptimizer:
    def __init__(self, frequencies: np.ndarray, lr: float = 0.001, betas: Tuple[float, float] = (0.9, 0.999), eps: float = 1e-8):
        self.frequencies = frequencies  # Parameters to optimize
        self.lr = lr          # Learning rate
        self.beta1, self.beta2 = betas  # Exponential decay rates for the moment estimates
        self.eps = eps        # Small value to prevent division by zero
        
        # Initialize first moment vector (mean of the gradient)
        #self.m = [np.zeros_like(freq) for freq in frequencies]
        self.m = np.zeros(frequencies.shape)

        # Initialize second moment vector (uncentered variance of the gradient)
        #self.v = [np.zeros_like(freq) for freq in frequencies]
        self.v = np.zeros(frequencies.shape)
        
        self.t = 0  # Initialize timestep

    def step(self, grads: np.ndarray) -> np.ndarray:
        self.t += 1
        lr_t = self.lr * np.sqrt(1 - self.beta2 ** self.t) / (1 - self.beta1 ** self.t)
        
        """
        for i in range(len(self.frequencies)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grads[i]
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grads[i] ** 2)
            
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            
            self.frequencies[i] -= lr_t * m_hat / (np.sqrt(v_hat) + self.eps)
        """
        
        self.m = self.beta1 * self.m + (1 - self.beta1) * grads
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grads ** 2)
        
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        
        self.frequencies -= lr_t * m_hat / (np.sqrt(v_hat) + self.eps)
        print(self.frequencies)
        self.frequencies /= np.sum(self.frequencies)
        print(self.frequencies)
        return self.frequencies