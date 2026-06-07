import numpy as np
from scipy.interpolate import RegularGridInterpolator
from core.base import InverseProblemWithLatent


class RotationProjection2D(InverseProblemWithLatent):

    def __init__(self, sigma, angles, X_grid, Y_grid, x_1d):
        super().__init__(sigma, angles)

        self.X = X_grid
        self.Y = Y_grid
        self.x_1d = x_1d

        self.shape_2d = X_grid.shape
        self.p = len(x_1d)

        # masque disque
        self.mask = (X_grid**2 + Y_grid**2) <= 1

        # coordonnées plates
        self.x_flat = X_grid.flatten()
        self.y_flat = Y_grid.flatten()

    # ===============================
    # ROTATION
    # ===============================
    def _rotate_array(self, array_2d, angle):
        """
        Rotation d'une fonction 2D via interpolation
        """

        cos_a = np.cos(-angle)
        sin_a = np.sin(-angle)

        # coordonnées transformées
        x_rot = cos_a * self.x_flat - sin_a * self.y_flat
        y_rot = sin_a * self.x_flat + cos_a * self.y_flat

        #  IMPORTANT : utiliser les axes 1D
        interp = RegularGridInterpolator(
            (self.x_1d, self.x_1d),
            array_2d,
            bounds_error=False,
            fill_value=0
        )

        pts = np.column_stack([x_rot, y_rot])
        rotated = interp(pts)

        return rotated.reshape(self.shape_2d)

    # ===============================
    # PROJECTION
    # ===============================
    def _project(self, array_2d):
        """
        Projection selon y :
        ∫ θ(x,y) dy
        """
        dy = self.x_1d[1] - self.x_1d[0]
        return np.trapz(array_2d, dx=dy, axis=0)

    # ===============================
    # OPÉRATEUR DIRECT
    # ===============================
    def apply_operator(self, theta_2d, angle):
        rotated = self._rotate_array(theta_2d, angle)
        return self._project(rotated)

    # ===============================
    # ADJOINT (approximation)
    # ===============================
    def apply_adjoint(self, y_1d, angle):
        """
        y_1d peut être :
        - (p,)     =>  1 signal
        - (n, p)    => batch de signaux
        """

        # cas batch
        if y_1d.ndim == 2:
            results = []
            for i in range(y_1d.shape[0]):
                results.append(self.apply_adjoint(y_1d[i], angle))
            return np.array(results)

        # cas simple (p,)
        retro = np.outer(np.ones(self.shape_2d[0]), y_1d)
        return self._rotate_array(retro, -angle)
    # ===============================
    # CONTRAINTE
    # ===============================
    def enforce_constraints(self, theta_2d):
        theta_2d = theta_2d.copy()
        theta_2d[~self.mask] = 0
        return theta_2d
