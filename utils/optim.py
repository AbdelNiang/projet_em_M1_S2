import numpy as np


class GradientAscent:
    def __init__(self, lr=0.1):
        self.lr = lr

    def step(self, theta, Q_fn, grad_fn):
        grad = grad_fn(theta)
        return theta + self.lr * grad


class ArmijoOptimizer:
    def __init__(self, eta0=1.0, beta=0.5, c=1e-4, max_iter=50):
        self.eta0 = eta0
        self.beta = beta
        self.c = c
        self.max_iter = max_iter

    def step(self, theta, Q_fn, grad_fn):
        grad = grad_fn(theta)
        eta = self.eta0
        Q_current = Q_fn(theta)

        for _ in range(self.max_iter):
            theta_new = theta + eta * grad
            Q_new = Q_fn(theta_new)

            if Q_new >= Q_current + self.c * eta * np.dot(grad.flatten(), grad.flatten()):
                return theta_new

            eta *= self.beta

        return theta