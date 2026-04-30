import numpy as np
import matplotlib.pyplot as plt

# ===== Models =====
from models.translation1d import Translation1D
from models.rotation2d import RotationProjection2D

# ===== EM =====
from core.em import em

# ===== Optim =====
from utils.optim import GradientAscent, ArmijoOptimizer

# ===== Utils =====
from utils.data import generate_data
from utils.metrics import translation_invariant_error
from utils.viz import (
    plot_signals,
    plot_samples_1d,
    plot_images,
    plot_convergence
)


# ============================================================
# CAS 1 : Translation 1D
# ============================================================

def demo_translation():
    print("\n" + "="*50)
    print("CAS 1 : Translation 1D")
    print("="*50)

    p = 50
    sigma = 0.3
    shifts = np.arange(-5, 6)

    model = Translation1D(sigma=sigma, shifts=shifts, p=p)

    x = np.linspace(0, 1, p)
    theta_true = np.exp(-50 * (x - 0.5)**2)

    Y, Z = generate_data(model, theta_true, n_samples=30)

    # 🔥 affichage des données
    plot_samples_1d(Y)

    theta_init = np.random.randn(p)

    theta_est, history, Q_history = em(
        model,
        Y,
        theta_init,
        n_iter=50
    )

    err = translation_invariant_error(theta_est, theta_true)
    print(f"Erreur (invariante translation) : {err:.4f}")

    plot_signals(theta_true, theta_est)
    plot_convergence(Q_history, "Convergence de Q (Translation)")


# ============================================================
# CAS 2 : Rotation + Projection 2D
# ============================================================

def demo_rotation():
    print("\n" + "="*50)
    print("CAS 2 : Rotation + Projection 2D")
    print("="*50)

    p = 40
    sigma = 0.3
    angles = np.linspace(0, 2*np.pi, 12, endpoint=False)

    x = np.linspace(-1, 1, p)
    X, Y_grid = np.meshgrid(x, x)

    model = RotationProjection2D(
        sigma=sigma,
        angles=angles,
        X_grid=X,
        Y_grid=Y_grid,
        x_1d=x
    )

    theta_true = np.exp(-8 * (X**2 + Y_grid**2))
    theta_true = model.enforce_constraints(theta_true)

    Y, Z = generate_data(model, theta_true, n_samples=10)

    # ⚠️ IMPORTANT : ici Y est 1D → pas imshow
    plot_samples_1d(Y)

    theta_init = np.random.randn(p, p)

    optimizer = ArmijoOptimizer()
    print("Optimizer utilisé :", optimizer.__class__.__name__)

    theta_est, history, Q_history = em(
        model,
        Y,
        theta_init,
        n_iter=30,
        optimizer=optimizer
    )

    theta_est = model.enforce_constraints(theta_est)

    err = np.linalg.norm(theta_est - theta_true) / np.linalg.norm(theta_true)
    print(f"Erreur relative : {err:.4f}")

    # affichage reconstruction 2D
    plot_images(theta_true, theta_est)

    # convergence
    plot_convergence(Q_history, "Convergence de Q (Rotation)")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    demo_translation()
    demo_rotation()