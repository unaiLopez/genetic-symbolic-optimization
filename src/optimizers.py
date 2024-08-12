import numpy as np

class AdamOptimizer:
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8):
        self.params = params  # Parameters to optimize
        self.lr = lr          # Learning rate
        self.beta1, self.beta2 = betas  # Exponential decay rates for the moment estimates
        self.eps = eps        # Small value to prevent division by zero
        
        # Initialize first moment vector (mean of the gradient)
        self.m = [np.zeros_like(p) for p in params]
        
        # Initialize second moment vector (uncentered variance of the gradient)
        self.v = [np.zeros_like(p) for p in params]
        
        self.t = 0  # Initialize timestep

    def step(self, grads):
        self.t += 1
        lr_t = self.lr * np.sqrt(1 - self.beta2 ** self.t) / (1 - self.beta1 ** self.t)

        for i, param in enumerate(self.params):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grads[i]
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grads[i] ** 2)
            
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            
            param -= lr_t * m_hat / (np.sqrt(v_hat) + self.eps)

# Example usage
# Initialize parameters (weights) and gradients
params = [np.random.randn(3, 3), np.random.randn(3, 1)]
grads = [np.random.randn(3, 3), np.random.randn(3, 1)]

# Initialize Adam optimizer
adam = AdamOptimizer(params, lr=0.001)

# Perform a single optimization step
adam.step(grads)

# Updated parameters
print("Updated Parameters:")
for p in params:
    print(p)