import numpy as np
from core.base import InverseProblemWithLatent

class Translation1D(InverseProblemWithLatent):
    """
    Cas 1 : (A_z θ)_j = θ(x_j - z) avec translation circulaire
    """

    def __init__(self, sigma, shifts, p):
        super().__init__(sigma, shifts)
        self.p = p
        self.is_isometry = True  #  important pour accélérer le M-step

    def apply_operator(self, theta, z):
        shift_int = int(np.round(z))
        return np.roll(theta, shift_int)

    def apply_adjoint(self, y, z):
        shift_int = int(np.round(z))
        return np.roll(y, -shift_int)
