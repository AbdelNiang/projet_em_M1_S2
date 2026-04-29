import numpy as np
from abc import ABC, abstractmethod


class InverseProblemWithLatent(ABC):

    def __init__(self, sigma, latent_grid, quad_weights=None):
        self.sigma = sigma
        self.latent_grid = np.asarray(latent_grid)
        self.n_latent = len(self.latent_grid)

        if quad_weights is None:
            self.quad_weights = np.ones(self.n_latent) / self.n_latent
        else:
            self.quad_weights = np.asarray(quad_weights)

    # ===============================
    # Opérateurs
    # ===============================

    @abstractmethod
    def apply_operator(self, theta, z):
        pass

    @abstractmethod
    def apply_adjoint(self, y, z):
        pass

    # ===============================
    # E-step
    # ===============================

    def compute_weights(self, Y, theta):

        AZ_theta = np.stack([
            self.apply_operator(theta, z)
            for z in self.latent_grid
        ])  # (K, p)

        diff = Y[:, None, :] - AZ_theta[None, :, :]
        losses = np.sum(diff**2, axis=2)

        exponents = -losses / (2 * self.sigma**2)

        # quadrature si non uniforme
        if self.quad_weights is not None:
            if not np.allclose(self.quad_weights, self.quad_weights[0]):
                exponents += np.log(self.quad_weights)[None, :]

        # stabilisation
        exponents -= np.max(exponents, axis=1, keepdims=True)

        weights = np.exp(exponents)
        weights /= np.sum(weights, axis=1, keepdims=True)

        return weights

    # ===============================
    # Gradient (GEM)
    # ===============================

    def gradient_Q(self, Y, theta):

        weights = self.compute_weights(Y, theta)

        AZ_theta = np.stack([
            self.apply_operator(theta, z)
            for z in self.latent_grid
        ])

        residuals = Y[:, None, :] - AZ_theta[None, :, :]

        A_adj_res = np.stack([
            self.apply_adjoint(residuals[:, k, :], z)
            for k, z in enumerate(self.latent_grid)
        ])  # (K, n, p)

        weights_expanded = weights.T[:, :, None]

        grad = np.sum(weights_expanded * A_adj_res, axis=(0, 1))

        return grad / (self.sigma**2)

    # ===============================
    # M-step explicite
    # ===============================

    def solve_m_step(self, Y, weights, lambda_reg=1e-6):

        n, p = Y.shape

        # ----- b -----
        b = np.zeros(p)

        for k, z in enumerate(self.latent_grid):
            A_adj_Y = self.apply_adjoint(Y, z)
            b += np.sum(weights[:, k][:, None] * A_adj_Y, axis=0)

        # ----- cas isométrique -----
        if hasattr(self, "is_isometry") and self.is_isometry:
            return b / n

        # ----- M -----
        M = np.zeros((p, p))

        for k, z in enumerate(self.latent_grid):
            for j in range(p):
                e_j = np.zeros(p)
                e_j[j] = 1

                Az_ej = self.apply_operator(e_j, z)
                A_adj_Az_ej = self.apply_adjoint(Az_ej, z)

                M[:, j] += np.sum(weights[:, k]) * A_adj_Az_ej

        M += lambda_reg * np.eye(p)

        # résolution
        try:
            theta = np.linalg.solve(M, b)
        except np.linalg.LinAlgError:
            theta = np.linalg.pinv(M) @ b

        return theta